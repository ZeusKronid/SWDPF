class Positioner: 
    
    def __init__(self
                 , forecast_path
                 , vessel_index
                 , datetime
                 , vessel_features =  [
                     
                    'wind_direction_true'
                     , 'wind_speed'
                     , 'sea_level_pressure'
                     , 'air_temperature'
                     #, 'sea_surface_temp' #TODO later
                     ,'wave_direction'
                     , 'wave_period'
                     , 'wave_height'
                     , 'swell_direction'
                     ,'swell_period'
                     , 'swell_height',
                     
                                        ]
                , forecast_features =  ['sp', 'gust','t', 'wind_direction'] # , 
                 , initial_lat = None
                 , initial_long = None ):
        self.forecast_path = forecast_path
        self.vessel_index = vessel_index
        
        self.forecast_features = forecast_features
        self.vessel_features = vessel_features

        self.initial_lat = initial_lat
        self.initial_long = initial_long
        self.datetime = datetime 

    def get_vessel_vector(self,):
        vessels = pd.read_csv('vessels.csv')
        my_vessel = vessels.iloc[self.vessel_index]
        self.initial_lat = my_vessel['latitude']
        self.initial_long = my_vessel['longitude']
        
        vector_of_parameters = pd.Series()
        for feature in self.vessel_features:
            vector_of_parameters[feature] = my_vessel[feature]
            
        vector_of_parameters['wave_direction'] = vector_of_parameters['swell_direction']
        vector_of_parameters = self.keep_uniform_naming(vector_of_parameters)
        return vector_of_parameters
        
    def get_vessel_boundaries(self, ): 
        
        min_lat = self.initial_lat - 1
        max_lat = self.initial_lat + 1
        min_lon = self.initial_long - 1
        max_lon = self.initial_long + 1
        
        return min_lat, max_lat, min_lon, max_lon 
        
    def longitude_convert(self, df):
        # Преобразуем значения longitude
        def map_function(lon):
            return (lon - 360) if lon > 180 else lon
        
        # Создаем новую колонку для преобразованных значений longitude
        df_reset = df.reset_index()
        df_reset['longitude'] = df_reset['longitude'].map(map_function)
        
        # Создаем новый MultiIndex с преобразованными значениями
        new_index = pd.MultiIndex.from_frame(df_reset[['latitude', 'longitude']])
        
        # Создаем новый DataFrame с новым MultiIndex
        df_new = df_reset.drop(columns=['latitude', 'longitude'])
        df_new.index = new_index
        
        return df_new

    
    def get_atmosphere_parameters(self):
        ds = xr.open_dataset(self.forecast_path, engine="cfgrib", filter_by_keys={'typeOfLevel': 'surface'})
        df = ds.to_dataframe()  # Convert xarray Dataset to pandas DataFrame
        df = self.longitude_convert(df)
        
        df_atmosphere = pd.DataFrame()
        
        df['wind_direction'] = self.calculate_wind_direction(df['utaua'], df['vtaua']) 
        
        for feature in self.forecast_features:
            df_atmosphere[feature] = df[feature]
        
        return df_atmosphere
        
    def get_wave_parameters(self,):
        min_lat, max_lat, min_lon, max_lon  = self.get_vessel_boundaries()
        
        wave_df = copernicusmarine.read_dataframe(
                dataset_id = "cmems_mod_glo_wav_my_0.2deg_PT3H-i",
                minimum_longitude = min_lon,
                maximum_longitude = max_lon,
                minimum_latitude = min_lat,
                maximum_latitude = max_lat,
                variables = ["VMDR", "VTPK", "VHM0", "VMDR_SW1", "VTM01_SW1", "VHM0_SW1"],
                start_datetime = self.datetime,
                end_datetime = self.datetime,
                username = "Stephen",
                password = "evtyzytn",
        )
        
        wave_df = wave_df.droplevel('time')
        wave_df = wave_df.sort_index(level=[0, 1], ascending=[False, True])
        wave_df.index = wave_df.index.map(lambda x: (round(x[0], 3), round(x[1], 3)))

        wave_df = self.longitude_convert(wave_df)

        return wave_df

    def get_water_temp(self, ):
        min_lat, max_lat, min_lon, max_lon  = self.get_vessel_boundaries()

        sea_temp_df = copernicusmarine.read_dataframe(
            dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
            minimum_longitude = min_lon,
            maximum_longitude = max_lon,
            minimum_latitude = min_lat,
            maximum_latitude = max_lat,
            variables = ["thetao",],
            start_datetime = self.datetime[:10],
            end_datetime = self.datetime[:10],
            username = "Stephen",
            password = "evtyzytn",
        )
        sea_temp_df = sea_temp_df.droplevel('time')
        sea_temp_df = sea_temp_df.sort_index(level=[0, 1], ascending=[False, True])
        sea_temp_df.index = sea_temp_df.index.map(lambda x: (round(x[0], 3), round(x[1], 3)))\
        
        sea_temp_df = self.longitude_convert(sea_temp_df)
        return sea_temp_df


    def interpolate_dataframes(self, df):
            # Получаем исходные координаты (latitude и longitude) из индекса

            # Получаем исходные координаты (latitude и longitude) из индекса
            old_coords = np.array(list(df.index))
        
            # Определяем диапазоны старых широт и долгот
            lat_min, lat_max = old_coords[:, 0].min(), old_coords[:, 0].max()
            lon_min, lon_max = old_coords[:, 1].min(), old_coords[:, 1].max()
        
            # Создаем сетку новых координат с шагом 0.01 для широты и долготы
            new_lats = np.round(np.arange(lat_min, lat_max + 0.01, 0.01), 2)
            new_lons =  np.round(np.arange(lon_min, lon_max + 0.01, 0.01), 2)
        
            # Создаем сетку координат
            new_coords = np.array(np.meshgrid(new_lats, new_lons)).T.reshape(-1, 2)
        
            # Разворачиваем DataFrame в массив значений
            values = df.values
        
            # Интерполяция данных на новую сетку координат
            interpolated_values = griddata(old_coords, values, new_coords, method='cubic')
        
            # Создаем новый MultiIndex с новыми координатами
            new_index = pd.MultiIndex.from_arrays([new_coords[:, 0], new_coords[:, 1]], names=['latitude', 'longitude'])
        
            # Создаем новый DataFrame с интерполированными значениями
            df_interpolated = pd.DataFrame(interpolated_values, index=new_index, columns=df.columns)
            
            return df_interpolated

        

    def calculate_wind_direction(self, tau_y, tau_x):
        # Calculate the angle in radians
        theta_rad = np.arctan2(tau_y, tau_x)
        
        # Convert radians to degrees
        theta_deg = np.degrees(theta_rad)
        
        # Normalize to range [0, 360)
        #wind_direction = (theta_deg + 180) % 360
        wind_direction = (theta_deg + 360) % 360        
        
        return wind_direction


    def keep_vessel_units(self, matrix_of_parameters):
        matrix_of_parameters['sp'] = matrix_of_parameters['sp']/100
        matrix_of_parameters['t'] = matrix_of_parameters['t'] - 273.15
        return matrix_of_parameters

    
        
    def get_bounded_atmosphere_matrix(self):
        min_lat, max_lat, min_lon, max_lon = self.get_vessel_boundaries()
        df = self.get_atmosphere_parameters()
    
        # Создаем булевы маски для фильтрации по latitude и longitude
        lat_filter = (df.index.get_level_values('latitude') >= min_lat) & (df.index.get_level_values('latitude') <= max_lat)
        lon_filter = (df.index.get_level_values('longitude') >= min_lon) & (df.index.get_level_values('longitude') <= max_lon)
        
        # Применяем фильтры к DataFrame
        result = df[lat_filter & lon_filter]

        

        matrix_of_parameters = self.keep_vessel_units(result)

        return matrix_of_parameters


    def get_interpolated_atmosphere_matrix(self,): 
        df = self.get_bounded_atmosphere_matrix()
        df_interpol = self.interpolate_dataframes(df)
        return df_interpol 
        
    def get_interpolated_wave_matrix(self,):
        df = self.get_wave_parameters()
        df_interpol = self.interpolate_dataframes(df)
        return df_interpol 
        
    def get_interpolated_water_temp_matrix(self,): 
        df = self.get_water_temp()
        df_interpol = self.interpolate_dataframes(df)
        return df_interpol 

    
    def get_final_forecast_matrix(self,): 
        atm = self.get_interpolated_atmosphere_matrix()
        #wave = self.get_interpolated_wave_matrix()
        #TODO: when have water params
        #water = self.get_interpolated_water_temp_matrix()
        #df = pd.concat([atm,wave],axis=1, join='inner')
        df = pd.concat([atm,  ], axis=1, join='inner')
        return df

    
    def keep_uniform_naming(self, vessel_vector): 
        vessel_vector = vessel_vector.rename({
            'wind_direction_true':'wind_direction'
            , 'wind_speed': 'gust'
            , 'sea_level_pressure': 'sp'
            , 'air_temperature' : 't'
            ,'wave_direction' : 'VMDR'
            , 'wave_period' : 'VTPK'
            , 'wave_height' : 'VHM0'
            , 'swell_direction' : 'VMDR_SW1'
            ,'swell_period' : 'VTM01_SW1'
            , 'swell_height' : 'VHM0_SW1'
            ,})
        vessel_vector = vessel_vector.reindex(self.forecast_features + [ 'VMDR', 'VTPK', 'VHM0', 'VMDR_SW1', 'VTM01_SW1', 'VHM0_SW1', ]) 
        return vessel_vector

    


    def plot_map(self, forecast_matrix, parameter_to_heat, observed_lat, oberved_long):
        
        lons = np.linspace(forecast_matrix.index.get_level_values('longitude').min(),
                   forecast_matrix.index.get_level_values('longitude').max(),
                   len(forecast_matrix.index.get_level_values('longitude').unique()))

        lats = np.linspace(forecast_matrix.index.get_level_values('latitude').min(),
                           forecast_matrix.index.get_level_values('latitude').max(),
                           len(forecast_matrix.index.get_level_values('latitude').unique()))
        
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        data_values = forecast_matrix[parameter_to_heat].values.reshape(len(lats), len(lons))  # Example random data for heatmap
        
        # Plotting
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())  # Use PlateCarree for unprojected lat/lon grid
        
        # Add map features (like coastlines)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Create a heatmap using pcolormesh
        heatmap = ax.pcolormesh(lon_grid, lat_grid, data_values, transform=ccrs.PlateCarree(), cmap='viridis')
        
        # Add a colorbar
        plt.colorbar(heatmap, ax=ax, orientation='vertical', label=parameter_to_heat)
        
        # Set the title
        plt.title("Initial and observed positions")
        
        
        ax.plot(self.initial_long, self.initial_lat, marker='o', color='white', markersize=5, transform=ccrs.PlateCarree())
        ax.plot(oberved_long, observed_lat, marker='o', color='red', markersize=8, transform=ccrs.PlateCarree())

        # === Настраиваем координатные оси ===
        # Задаем пределы осей (расширение карты)
        #ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
        # Добавляем сетку с координатами
        grid_lines = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=1, color='gray', alpha=0.5, linestyle='--')
        grid_lines.top_labels = False  # Отключаем метки сверху
        grid_lines.right_labels = False  # Отключаем метки справа
        grid_lines.xformatter = LongitudeFormatter()  # Форматируем долготу
        grid_lines.yformatter = LatitudeFormatter()  # Форматируем широту
        grid_lines.xlabel_style = {'size': 10, 'color': 'black'}
        grid_lines.ylabel_style = {'size': 10, 'color': 'black'}
        
        
        # Show the plot
        plt.show()

    def find_distances(self, subtract_matrix): 
        distances = np.linalg.norm(subtract_matrix, axis=1)
        # Находим индекс ближайшего вектора
        subtract_matrix['distances'] = distances
        # Извлекаем ближайший вектор
        return subtract_matrix

    def get_observed_lat_long(self, ):
        vessel_vector = self.get_vessel_vector()
        forecast_matrix = self.get_final_forecast_matrix()
        vessel_vector = vessel_vector.drop([ 'VMDR', 'VTPK', 'VHM0', 'VMDR_SW1', 'VTM01_SW1', 'VHM0_SW1', ])
        print(forecast_matrix)
        print(vessel_vector)
        subtraction = forecast_matrix.sub(vessel_vector)
        subtraction_with_distances = self.find_distances(subtraction)
        
        position_of_minimum  = subtraction_with_distances[subtraction_with_distances['distances'] == min(subtraction_with_distances['distances'])].index
        latitude, longitude = position_of_minimum[0]

        self.plot_map(forecast_matrix, 't',latitude, longitude)
        print(round(latitude-self.initial_lat, 5), round(longitude-self.initial_long,3))
        return (latitude, longitude), (self.initial_lat, self.initial_long)