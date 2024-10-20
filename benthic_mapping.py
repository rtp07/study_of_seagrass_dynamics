import xarray as xr
import rasterio
import pandas as pd
import numpy as np
import rasterio.transform
from datetime import datetime
from rasterio.features import rasterize
from scipy.stats import linregress

def mask_dataset(dataset, geodataframe, desired_year=None, target_epsg=32748):
    """
    Mask a dataset based on polygons from a shapefile for a specified year.

    Args:
        dataset (xarray.Dataset): The dataset to be masked.
        geodataframe (geopandas.GeoDataFrame): The GeoDataFrame containing polygons.
        desired_year (int, optional): User-defined year for time filtering. Defaults to None.
        target_epsg (int, optional): EPSG code for the target crs. Defaults to 32748.

    Returns:
        xarray.Dataset: The masked dataset.
    """
    
    # Ensure geodataframe is provided
    if geodataframe is None:
        raise ValueError("A GeoDataFrame must be provided.")

    # Filter the GeoDataFrame for the desired year, if provided
    if desired_year is not None:
        geodataframe['date'] = pd.to_datetime(geodataframe['date'])
        geodataframe['year'] = geodataframe['date'].dt.year
        geodataframe = geodataframe[geodataframe['year'] == desired_year]

        # Ensure data exists for the desired year
        if geodataframe.empty:
            raise ValueError(f"No data found for the year {desired_year}. Please check your input.")

    # Convert geometry to the target EPSG code if not already
    if geodataframe.crs is None:
        raise ValueError("The GeoDataFrame must have a CRS defined.")
    if geodataframe.crs.to_epsg() != target_epsg:
        print(f"Converting geometry from EPSG:{geodataframe.crs.to_epsg()} to EPSG:{target_epsg}")
        geodataframe = geodataframe.to_crs(epsg=target_epsg)

    # Define bounds and create a transformation
    bounds = (dataset.x.min().item(), dataset.y.min().item(), dataset.x.max().item(), dataset.y.max().item())
    transform = rasterio.transform.from_bounds(*bounds, dataset.sizes['x'], dataset.sizes['y'])
    
    # Rasterize the geometries from the GeoDataFrame to create a binary mask
    object_mask = rasterize(
        [(geom, 1) for geom in geodataframe.geometry],
        out_shape=(dataset.sizes['y'], dataset.sizes['x']),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )

    # Mask the dataset
    masked_ds = dataset.where(object_mask == 1)

    return masked_ds

def sunglint_correction(dataset, samples, nir_band_name, vars_ignore=None):
    """
    Funtion to perform sunglint correction based on NIR brightness (Hedley at al., 2005).

    Args:
        dataset (xarray.Dataset): The dataset containing the original surface reflectance data to be corrected. 
        samples (xarray.Dataset): A subset of deep water samples from the dataset.
        nir_band_name (str): The name of the NIR band variable in the dataset
        vars_ignore (list of str, optional): A list of variable names to ignore during the correction. Defaults to None.

    Returns:
        xarray.Dataset: The corrected dataset.
    """
    if vars_ignore is None:
        vars_ignore = []

    # Extract NIR band data from samples
    nir_data = samples[nir_band_name].values.flatten()
    MinNir = np.nanmin(nir_data)
    print(f'Minimum NIR brightness (MinNir): {MinNir}')
    
    # Initialize dictionaries to store results
    slopes = {}
    
    # Iterate over each variable in the samples
    for var_name, var_data in samples.data_vars.items():
        # Skip the NIR band and any variables in vars_ignore
        if var_name == nir_band_name or var_name in vars_ignore:
            continue
        
        # Extract and flatten the data for the current variable
        var_data_array = var_data.values.flatten()
        
        # Remove NaNs to avoid errors in regression
        valid_mask = ~np.isnan(nir_data) & ~np.isnan(var_data_array)
        if np.sum(valid_mask) < 2:  # Need at least 2 points for regression
            print(f"Not enough valid data for regression with variable '{var_name}'.")
            continue
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(nir_data[valid_mask], var_data_array[valid_mask])
        
        # Store the results
        slopes[var_name] = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err
        }
        
        print(f"Regression results for {var_name}: slope={slope}, r_value={r_value}, p_value={p_value}")
    
    corrected_data_vars = {}
    for var_name, var_data in dataset.data_vars.items():
        if var_name == nir_band_name or var_name in vars_ignore:
            continue
        
        slope_info = slopes.get(var_name, None)
        if slope_info is None:
            print(f"Slope information not found for variable '{var_name}'. Skipping correction.")
            continue
        
        # Unpack the slope information
        slope = slope_info['slope']
        intercept = slope_info['intercept']
        r_value = slope_info['r_value']
        p_value = slope_info['p_value']
        std_err = slope_info['std_err']
        
        # Apply the correction
        data_array = var_data.values
        corrected_data_array = data_array - slope * (dataset[nir_band_name].values - MinNir)
        
        # Create a new DataArray with corrected data
        var_attrs = var_data.attrs.copy()
        var_attrs['wave_nm'] = var_attrs.get('wave_nm', 'Unknown')
        
        # Handle variable names with or without underscores
        long_name = f'Sun Glint Corrected Surface Reflectance'
     
        # Assign the corrected DataArray with a new name
        new_name = f"{var_name}_sg"
        corrected_data_vars[new_name] = xr.DataArray(
            data=corrected_data_array,
            dims=dataset[var_name].dims,
            coords=dataset[var_name].coords,
            name=new_name,
            attrs={
                'wave_nm': var_attrs['wave_nm'],
                'long_name': long_name,
                'units': '1',
                'min_nir_value': MinNir,
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err,
                'history': 'Sun glint correction was applied using NIR-based linear regression.',
                'date_created': datetime.utcnow().isoformat(),
            }
        )

    # Create a new dataset with corrected data variables
    corrected_dataset = xr.Dataset(
        data_vars=corrected_data_vars,
        attrs=dataset.attrs.copy()
    )

    return corrected_dataset

def calculate_k_ratio(dataset, var1_name, var2_name):
    """
    Calculate the attenuation ratio k_i/k_j between two variables. 
    Ideally, the dataset should come from a subset of an object (e.g., sand),
    collected at different depths (Green et al., 2000).

    Args:
        dataset (xarray.Dataset): The dataset containing the reflectance values.
        var1_name (str): The name of the first variable.
        var2_name (str): The name of the second variable.

    Returns:
        tuple: A tuple containing k_ratio, variance_Bi, variance_Bj, and covariance_BiBj.
    """

    # Access reflectance values
    B_i = dataset[var1_name]
    B_j = dataset[var2_name]
        
    # Remove NaN values from both arrays for accurate calculations
    valid_mask = ~np.isnan(B_i) & ~np.isnan(B_j)
    B_i_valid = B_i.where(valid_mask, drop=True)
    B_j_valid = B_j.where(valid_mask, drop=True)
    
    # Calculate variance
    variance_Bi = B_i_valid.var(dim=['x', 'y'])
    variance_Bj = B_j_valid.var(dim=['x', 'y'])
    
    # Calculate covariance
    covariance_BiBj = xr.cov(B_i_valid, B_j_valid, dim=['x', 'y'])
    
    # Calculate 'a' using the given formula
    a = (variance_Bi - variance_Bj) / (2 * covariance_BiBj)
    
    # Calculate the ratio k_i/k_j
    k_ratio = a + np.sqrt(a**2 + 1)
    
    # Return all necessary values as a tuple
    return k_ratio.item(), variance_Bi.item(), variance_Bj.item(), covariance_BiBj.item()

def calculate_dii(dataset, var1_name, var2_name, k_ratio):
    """
    Calculate the Depth Invariant Index (DII) between two bands given a k-ratio.
    
    Args:
        dataset (xarray.Dataset): The dataset containing the reflectance values.
        var1_name (str): The name of the first band.
        var2_name (str): The name of the second band.
        k_ratio (float): The attenuation ratio k_i/k_j.
    
    Returns:
        xarray.DataArray: The calculated DII values.
        float: The offset added to ensure non-negative DII values.
    """
    # Access reflectance values as xarray.DataArrays
    B_i = dataset[var1_name]
    B_j = dataset[var2_name]

    # Ensure no non-positive values for log calculation
    B_i_valid = B_i.where(B_i > 0, drop=True)
    B_j_valid = B_j.where(B_j > 0, drop=True)

    # Align dimensions of B_i_valid and B_j_valid
    B_i_valid, B_j_valid = xr.align(B_i_valid, B_j_valid, join='inner')

    # Calculate the Depth Invariant Index (DII)
    dii = np.log(B_i_valid) - (k_ratio * np.log(B_j_valid))
    
    # Determine the minimum DII value and calculate offset if needed
    min_dii = dii.min().values
    offset = 0.0
    if min_dii < 0:
        offset = -min_dii  # Offset to make all DII values non-negative
        dii = dii + offset
        
    # Create a DataArray with the same coordinates and dimensions as the input bands
    dii_data_array = xr.DataArray(
        data=dii.values,
        dims=B_i_valid.dims,
        coords=B_i_valid.coords
    )

    return dii_data_array, offset


def water_column_correction(dataset, samples, band_pairs):
    """
    Corrects water column effects using Depth Invariant Index (DII).

    Args:
        dataset (xarray.Dataset): Dataset with variables (bands) for DII calculation.
        samples (xarray.Dataset): Subset used to compute the k-ratio for correction.
        band_pairs (list of tuples): Pairs of band names.

    Returns:
        xarray.Dataset: Water column corrected dataset.
    """
    corrected_dataset = xr.Dataset(
        attrs=dataset.attrs.copy()
    )
    
    # Iterate over each band pair, calculate k_ratio, and then calculate DII
    for var1_name, var2_name in band_pairs:
        k_ratio, variance_Bi, variance_Bj, covariance_BiBj = calculate_k_ratio(samples, var1_name, var2_name)
        key = f'dii_{var1_name}_{var2_name}'  
        print(f'Calculating DII for bands {var1_name} and {var2_name} with k-ratio: {k_ratio}')
        dii_values, offset = calculate_dii(dataset, var1_name, var2_name, k_ratio)

        valid_range = (np.nanmin(dii_values), np.nanmax(dii_values))
        
        # Update attributes of the DataArray
        dii_values.attrs.update({
            'wave_nm_1': dataset[var1_name].attrs.get('wave_nm', 'Unknown'),
            'wave_nm_2': dataset[var2_name].attrs.get('wave_nm', 'Unknown'),
            'long_name': f"Depth Invariant Index from {var1_name.capitalize()} and {var2_name.capitalize()} paired bands",
            'units': '1',
            'offset': offset,
            'k_ratio': k_ratio,
            'valid_range': valid_range,
            'variance_Bi': variance_Bi,
            'variance_Bj': variance_Bj,
            'covariance_BiBj': covariance_BiBj,
            'history': 'Water column correction was applied',
            'date_created': datetime.utcnow().isoformat(),
        })

        # Add the DataArray to the dataset
        corrected_dataset[key] = dii_values
        
    return corrected_dataset

def normalized_difference(dataset, var_pairs, var_names):
    """
    Computes the normalized difference for each pair of variables in a dataset and returns a dataset with the results.

    Args:
        dataset (xarray.Dataset): Dataset containing the variables to be compared.
        var_pairs (list of tuples): List of tuples where each tuple contains the names of two variables to be compared.
        var_names (list of str): List of names for the resulting normalized difference variables.

    Returns:
        xarray.Dataset: Dataset containing the normalized differences for each pair of variables.
    """
    results = {}

    for (var1_name, var2_name), name in zip(var_pairs, var_names):
        # Compute the normalized difference
        nd_result = (dataset[var1_name] - dataset[var2_name]) / (dataset[var1_name] + dataset[var2_name])
        
        # Create a DataArray with the computed result
        nd_data_array = xr.DataArray(
            data=nd_result,
            dims=dataset[var1_name].dims,
            coords=dataset[var1_name].coords,
            name=name,
            attrs={
                'long_name': f"Normalized Difference between {var1_name.upper()} and {var2_name.upper()}",
                'units': '1',
                'date_created': datetime.utcnow().isoformat(),
            }
        )
        
        results[name] = nd_data_array
    
    return xr.Dataset(
        data_vars=results,
        attrs=dataset.attrs.copy()
    )

def labeled_samples(dataset, geodataframe, label, desired_year=None, target_epsg=32748):
    """
    Create a labeled data variables on the dataset by rasterizing geospatial data and merging it with the dataset.

    Args:
        dataset (xarray.Dataset): The dataset containing the data to be labeled.
        geodataframe (geopandas.GeoDataFrame): GeoDataFrame with geometries and labels.
        label (str): The column name in GeoDataframe to use as labels.
        desired_year (int, optional): The year of data to extract from GeoDataFrame. Defaults to None.
        target_epsg (int, optional): The EPSG code to reproject geometries. Defaults to 32748.

    Returns:
        xarray.Dataset: The masked dataset with an additional label data variable. 
    """
    
    # Ensure geodataframe and desired_year are provided
    if geodataframe is None:
        raise ValueError("A GeoDataFrame must be provided.")
    # Filter GeoDataFrame for the desired year if specified
    if desired_year is not None:
        geodataframe['date'] = pd.to_datetime(geodataframe['date'])
        geodataframe['year'] = geodataframe['date'].dt.year
        geodataframe = geodataframe[geodataframe['year'] == desired_year]
    
    # Convert geometry to the target EPSG code if not already
    if geodataframe.crs.to_epsg() != target_epsg:
        print(f"Converting geometry from EPSG:{geodataframe.crs.to_epsg()} to EPSG:{target_epsg}")
        geodataframe = geodataframe.to_crs(epsg=target_epsg)
        
    bounds = (dataset.x.min(), dataset.y.min(), dataset.x.max(), dataset.y.max())
    transform = rasterio.transform.from_bounds(*bounds, dataset.sizes['x'], dataset.sizes['y'])

    label_mask = rasterize(
        [(geom, class_value) for geom, class_value in zip(geodataframe.geometry, geodataframe[label])],
        out_shape=(dataset.sizes['y'], dataset.sizes['x']),
        transform=transform,
        fill=255,
        all_touched=True,
        dtype=float,
    )
    
    label_mask[label_mask == 255] = np.nan
    
    label_masked_ds = dataset.where(~np.isnan(label_mask))
    
    # Add the label mask as a new variable to the dataset
    label_masked_ds['label'] = xr.DataArray(
        data=label_mask,
        dims=dataset[list(dataset.data_vars)[0]].dims,
        coords=dataset[list(dataset.data_vars)[0]].coords,
        attrs={
            'long_name': 'Label DataArray',
            'data_created': datetime.utcnow().isoformat(),
        }
    )

    return label_masked_ds

def prepare_samples(samples, features, label_column: str):
    """
    Prepare feature matrix (X) and label vector (y) from the given samples.

    Args:
        samples (xarray.Dataset): The dataset containing the sample data.
        features (list): List of feature names to include in X.
        label_column (str): Name of the variable containing target labels

    Returns:
        nd.array: X and y array.
    """
    
    # Stact the feature data into a 2D array 'X'
    X = np.stack(
        [samples[var].values.flatten() for var in features], axis=1
    )
    
    # Extract the target labels and flatten into a 1D array 'y'
    y = samples[label_column].values.flatten()
    
    # Create a mask to filter out samples with NaN labels
    mask = ~np.isnan(y)
    
    # Apply the mask to both 'X' and 'y"
    X = X[mask]
    y = y[mask]
    
    # Round labels and convert them to integers
    y = np.round(y).astype(int)
    
    return X, y