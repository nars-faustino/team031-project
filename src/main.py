from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from bokeh.models.layouts import Row, Column, GridBox
    from bokeh.models.plots import GridPlot
    from bokeh.events import ButtonClick

import json
import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import (
    Select, Button, ColumnDataSource, Div, ImageURL, LabelSet, Patches, 
    ColorBar, CategoricalColorMapper, HoverTool, Range1d, FactorRange,
    TabPanel, Tabs
)
from bokeh.models.css import Styles
from bokeh.sampledata.us_states import data as states
from bokeh.palettes import RdYlGn6 as palette

import holoviews as hv
from holoviews.streams import Stream

hv.extension('bokeh')

import joblib
import datetime


# Custom streamer to update HoloViews DynamicMap plots
class UpdateStream(Stream):
    def event(self, **kwargs):
        super().event(**kwargs) # Triggers the update


class AirfarePredictionApp():

    def __init__(self) -> None:

        # Default Margins
        self.default_margins = (20, 20, 20, 20)

        # Load data into memory
        self.df = pd.read_csv(r'./data/processed-data.csv')

        # Load the saved XGBoost model pipeline
        self.xgb_model = joblib.load(r'./models/xgb_airfare_model.pkl')

        # Load the saved CatBoost model pipeline
        self.catboost_model = joblib.load(r'./models/catboost_airfare_model.pkl')               

        # ~ load saved Decision Tree
        self.decision_tree_model = joblib.load(r'./models/best_decision_tree_regressor.pkl')

        # ~ load saved Random Forest
        self.random_forest_model = joblib.load(r'./models/best_random_forest_regressor.pkl')

        # ~ load saved Decision Tree & Random Forest model columns
        self.decision_forest_model_columns = joblib.load(r'./models/decision_and_forest_model_columns.pkl')

        # Load Prophet Forecast DataFrames from JSON (dict of pd.Dataframe.to_dict(orient='records'))
        with open(r'./models/prophet_model_fare_forecast.json', 'r') as file_in:
            self.prophet_fare_dfs = json.load(file_in)
        with open(r'./models/prophet_model_farelg_forecast.json', 'r') as file_in:
            self.prophet_farelg_dfs = json.load(file_in)
        with open(r'./models/prophet_model_farelow_forecast.json', 'r') as file_in:
            self.prophet_farelow_dfs = json.load(file_in)

        # Load Prophet Forecast DataFrames (where routes had IATA code remapped)
        # note: after preprocessing data, no valid routes needed remapping, hence empty dicts below
        self.prophet_fare_dfs2 = {}
        self.prophet_farelg_dfs2 = {}
        self.prophet_farelow_dfs2 = {}

        # Load CSS
        with open(r'./src/styles.css') as f:
            self.css = f.read()

        # Inputs
        self.dropdown_origin = Select(
            title='Origin', 
            value='',
            options=[''] + sorted(list(self.df['airport_name_concat_1'].unique())), 
            margin=self.default_margins, 
            width=200
        )
        self.dropdown_origin.on_change(
            'value', self._handle_origin_input_change
        )

        self.dropdown_destination = Select(
            title='Destination', 
            value='',
            options=[''] + sorted(list(self.df['airport_name_concat_2'].unique())), 
            margin=self.default_margins, 
            width=200
        )
        self.dropdown_destination.on_change(
            'value', self._handle_destination_input_change
        )

        self.dropdown_season = Select(
            title='Season of Travel', 
            value='', 
            options = [
                '',
                'Spring',
                'Summer',
                'Fall',
                'Winter'
            ], 
            margin=self.default_margins, 
            width=200
        )
        self.dropdown_season.on_change(
            'value', self._handle_season_input_change
        )

        self.dropdown_ml_model = Select(
            title='ML Model Selection', 
            value='', 
            margin=(0, 20, 0, 20), 
            options = [
                '',
                'FB Prophet',
                'Random Forest',
                'Decision Tree',
                'XGBoost',
                'CatBoost'
            ],
            width=200
        )
        self.dropdown_ml_model.on_change(
            'value', self._handle_ml_model_input_change
        )

        self.button_run_analyzer = Button(
            label='Analyze', 
            button_type='success', 
            margin=self.default_margins,
            width=200
        )

        self.analysis_results = Div(
            text="Est. Price:<h2>$   -  </h2>", 
            height=50, 
            width=100, 
            margin=(0, 0, 0, 20), 
            stylesheets=[self.css]
        )

        # Misc variables
        self.EXCLUDED = ('HI', 'AK') # Excluded states
        self.multi_airport_codes = {'ACY', 'ORD', 'DTW', 'LGA', 'IAD', 'EGE'} # IATA codes with multiple airports after remapping
        self.ts_cols = ['year', 'quarter', 'fare', 'fare_lg', 'fare_low'] # time series columns

        # Calculate coordinates for all airports in dataset
        self.airport_coords = {
            row['airport_name_concat_1']: (row['longitude_1'], row['latitude_1']) 
            for _, row in 
                self.df[['airport_name_concat_1', 'longitude_1', 'latitude_1']]
                    .drop_duplicates()
                    .iterrows()
        }

        self.airport_coords.update({
            row['airport_name_concat_2']: (row['longitude_2'], row['latitude_2']) 
            for _, row in 
                self.df[['airport_name_concat_2', 'longitude_2', 'latitude_2']]
                    .drop_duplicates()
                    .iterrows()
        })

        # Calculate mapping between state abrvn & state name (ex. CA --> California)
        self.state_mapping = {k: v['name'] for k, v in states.items()}

        # Bar chart limits
        self.bar_chart_xlim = {}
        self.bar_chart_ylim = {}

        # Prophet Time Series Analysis Plotting
        self.prophet_fare_df = None  # store Prophet pd.DataFrame (history + forecast) to plot
        self.prophet_fare_lg_df = None
        self.prophet_fare_low_df = None

        return None


    # ----------------------------------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------------------------------
    def _update_analysis_results(self, value: float) -> None:
        """Updates analysis results text box
        """
        if value == 0:
            self.analysis_results.text = "Est. Price:<h2>$   -  </h2>"

        else:
            self.analysis_results.text = f'Est. Price:<h2>$ {value:.2f}</h2>'
        return None


    def _get_filtered_data(self, filter_season: bool = False) -> pd.DataFrame:
        """Returns a filtered dataframe based on the options selected
        """

        filtered_df = self.df.copy(deep=True)

        if self.dropdown_origin.value != '':
            filtered_df = filtered_df[
                filtered_df['airport_name_concat_1'] == self.dropdown_origin.value
            ]

        if self.dropdown_destination.value != '':
            filtered_df = filtered_df[
                filtered_df['airport_name_concat_2'] == self.dropdown_destination.value
            ]

        if filter_season:
            if self.dropdown_season.value != '':
                filtered_df = filtered_df[
                    filtered_df['season'] == self.dropdown_season.value
                ]

        return filtered_df
    

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['lat_diff'] = df['latitude_2'] - df['latitude_1']
        df['lon_diff'] = df['longitude_2'] - df['longitude_1']
        df['same_state'] = (df['state_1'] == df['state_2']).astype(int)
        df['route'] = df['airport_iata_1'] + "-" + df['airport_iata_2']
        df['distance_bin'] = pd.cut(
            df['nsmiles'],
            bins=[0, 500, 1500, 3000, 6000],
            labels=['short', 'medium', 'long', 'ultra']
        )
        return df


    # ----------------------------------------------------------------------------------------------
    # Choropleth Functions
    # ----------------------------------------------------------------------------------------------
    def _initialize_choropleth(self) -> None:
        """Function to initialize the choropleth chart. No data should be plotted yet.
        """

        # Figure for Choropleth
        self.choropleth = figure(
            title='Average Airfare Prices',
            height=400,
            width=800,
            x_axis_location=None, 
            y_axis_location=None,
            margin=self.default_margins
        )
        self.choropleth.grid.grid_line_color = None
        self.choropleth.toolbar.logo = None

        # State outlines
        state_xs = [states[code]["lons"] for code in states if code not in self.EXCLUDED]
        state_ys = [states[code]["lats"] for code in states if code not in self.EXCLUDED]
        self.choropleth_default_colors = ['#FFFFFF' for _ in range(len(state_xs))]
        self.choropleth_default_fares = ['-' for _ in range(len(state_xs))]
        
        self.choropleth_state_src = ColumnDataSource(dict(
            xs=state_xs,
            ys=state_ys,
            names=[self.state_mapping[code] for code in states if code not in self.EXCLUDED],
            abrvns=[code for code in states if code not in self.EXCLUDED],
            colors=self.choropleth_default_colors,
            avg_fares=self.choropleth_default_fares
        ))

        self.choropleth_patches = Patches(
            xs='xs', 
            ys='ys', 
            fill_color='colors',
            fill_alpha=0.7, 
            line_color='#000000', 
            line_alpha=0.3,
            line_width=2
        )
        self.choropleth.add_glyph(self.choropleth_state_src, self.choropleth_patches)        

        # Markers for origin / destination
        x0, y0 = self.airport_coords['SAN - San Diego International Airport']

        # Origin
        self.choropleth_origin_src = ColumnDataSource(dict(
            xs=[x0],
            ys=[y0],
            names=['Origin']
        ))

        self.choropleth_origin_marker = ImageURL(
            url={'value': 'https://cdn-icons-png.flaticon.com/128/18395/18395847.png'},
            x='xs',
            y='ys',
            w=2,
            h=2,
            global_alpha=0.0,
            anchor='center'
        )

        self.choropleth_origin_label = LabelSet(
            x='xs', 
            y='ys', 
            text='names',
            text_alpha=0.0,
            text_font_size='8pt',
            x_offset=-16,
            y_offset=-20, 
            source=self.choropleth_origin_src
        )

        self.choropleth.add_glyph(self.choropleth_origin_src, self.choropleth_origin_marker)
        self.choropleth.add_layout(self.choropleth_origin_label)

        # Destination
        self.choropleth_dest_src = ColumnDataSource(dict(
            xs=[x0],
            ys=[y0],
            names=['Destination']
        ))

        self.choropleth_dest_marker = ImageURL(
            url={'value': 'https://cdn-icons-png.flaticon.com/128/447/447031.png'},
            x='xs',
            y='ys',
            w=2,
            h=2,
            global_alpha=0.0,
            anchor='bottom'
        )

        self.choropleth_dest_label = LabelSet(
            x='xs', 
            y='ys', 
            text='names',
            text_alpha=0.0,
            text_font_size='8pt',
            x_offset=-28,
            y_offset=-14, 
            source=self.choropleth_dest_src
        )

        self.choropleth.add_glyph(self.choropleth_dest_src, self.choropleth_dest_marker)
        self.choropleth.add_layout(self.choropleth_dest_label)

        # Hover Tooltip
        hover = HoverTool(
            tooltips=[
                ("State", "@names"),
                ("Code", "@abrvns"),
                ("Avg. Fare", "@avg_fares")
            ], 
            mode='mouse'
        )
        self.choropleth.add_tools(hover)

        # Color Bar
        self.choropleth_default_mapper = CategoricalColorMapper(
            palette=['#D3D3D3'],
            factors=['$0.00']
        )

        self.choropleth_color_bar = ColorBar(
            color_mapper=self.choropleth_default_mapper,
            visible=True
        )
        
        self.choropleth.add_layout(self.choropleth_color_bar, 'below')

        return None


    def _update_choropleth(self) -> None:
        """Updates choropleth chart based on inputs
        """

        # Origin Value Updater
        origin_value = self.dropdown_origin.value
        if origin_value == '':
            self.choropleth_state_src.data['colors'] = self.choropleth_default_colors
            self.choropleth_state_src.data['avg_fares'] = self.choropleth_default_fares
            self.choropleth_origin_marker.global_alpha = 0.0
            self.choropleth_origin_label.text_alpha = 0.0
            self.choropleth_color_bar.update(color_mapper=self.choropleth_default_mapper)

        else:

            # Update origin marker
            x, y = self.airport_coords[origin_value]
            self.choropleth_origin_src.data['xs'] = [x]
            self.choropleth_origin_src.data['ys'] = [y]
            self.choropleth_origin_marker.global_alpha = 1.0
            self.choropleth_origin_label.text_alpha = 1.0

            # Filter / aggregate data
            df = (
                self.df[(self.df['airport_name_concat_1'] == origin_value)]
                    .groupby('state_2')
                    .agg({'fare': 'mean'})
            )

            fares = df.to_dict(orient='index')
            
            # Update state colors
            bins = np.linspace(
                df['fare'].min() - 0.001, # account for float roundoff error
                df['fare'].max() + 0.001, 
                num=7
            )

            d = pd.cut(
                df['fare'], 
                bins=bins, 
                labels=list(range(6))
            ).to_dict()

            if df.shape[0] > 0:

                # Calculate colors
                state_colors = []
                avg_fares = []

                for code in states:
                    if code not in self.EXCLUDED:
                        name = self.state_mapping[code]
                        
                        if name in d:
                            state_colors.append(palette[d[name]])
                            avg_fares.append(f"${fares[name]['fare']:.2f}")

                        else:
                            state_colors.append('#FFFFFF')
                            avg_fares.append('-')

                self.choropleth_state_src.data['colors'] = state_colors
                self.choropleth_state_src.data['avg_fares'] = avg_fares

                # Update colorbar
                new_color_mapper = CategoricalColorMapper(
                    palette=palette,
                    factors=[
                        f'${0.5*(bins[i] + bins[i+1]):.2f}' for i in range(6)
                    ]
                )
                self.choropleth_color_bar.update(color_mapper=new_color_mapper)

            else:

                # Hide / Reset if no data available
                self.choropleth_state_src.data['colors'] = self.choropleth_default_colors
                self.choropleth_state_src.data['avg_fares'] = self.choropleth_default_fares
                self.choropleth_color_bar.update(color_mapper=self.choropleth_default_mapper)


        # Destination Value Updater
        destination_value = self.dropdown_destination.value
        if destination_value == '':
            self.choropleth_dest_marker.global_alpha = 0.0
            self.choropleth_dest_label.text_alpha = 0.0

        else:
            # Update destination marker
            x, y = self.airport_coords[destination_value]
            self.choropleth_dest_src.data['xs'] = [x]
            self.choropleth_dest_src.data['ys'] = [y]
            self.choropleth_dest_marker.global_alpha = 1.0
            self.choropleth_dest_label.text_alpha = 1.0

        return None


    # ----------------------------------------------------------------------------------------------
    # Histogram Functions
    # ----------------------------------------------------------------------------------------------
    def _redraw_holoviews_histogram(self) -> hv.Histogram:
        """Function to redefine the histogram using the HoloViews API
        """

        # Tooltip
        hover_tooltips = [
            ('Frequency', '@top')
        ]
        
        # Initialize with invisible chart
        if any([self.dropdown_origin.value == '', self.dropdown_destination == '']):
            alpha = 0.
            
        else:
            alpha = 0.8

        # Get filtered data
        filtered_df = self._get_filtered_data(filter_season=True)

        # Calculate histogram
        bins = np.arange(filtered_df["fare"].min(), filtered_df["fare"].max() + 20, 20)
        hist, edges = np.histogram(filtered_df["fare"], bins=bins)

        # Update limits based on values
        self.histogram_xlim = Range1d(edges[0] - 20, edges[-1] + 20)
        self.histogram_ylim = Range1d(0, np.max(hist) + 20)

        return hv.Histogram((edges, hist)).opts(
            xlabel="Fare (Large Market Share)",
            ylabel="Frequency",
            title=f"Fare Distribution",
            width=800, 
            height=400, 
            tools=["hover"],
            color="#aec7e8", 
            line_color="black", 
            alpha=alpha,
            hover_tooltips=hover_tooltips,
            margin=self.default_margins
        )


    def _initialize_histogram(self) -> None:
        """Function to initialize the histogram charts.
        """
        self.hv_histogram = hv.DynamicMap(self._redraw_holoviews_histogram, streams=[UpdateStream()])
        self.bk_histogram = hv.render(self.hv_histogram)
        self.bk_histogram.toolbar.logo = None
        return None


    def _update_histogram(self) -> None:
        """Updates the histogram chart when new options are selected.
        """
        self.hv_histogram.event() # Trigger a redraw, recalculate xlim, ylim
        self.bk_histogram.x_range = self.histogram_xlim
        self.bk_histogram.y_range = self.histogram_ylim
        return None
    

    # ----------------------------------------------------------------------------------------------
    # Line Chart Functions
    # ----------------------------------------------------------------------------------------------
    def _redraw_holoviews_line_chart(self) -> hv.Overlay:
        """Function to redefine the line chart using the HoloViews API
        """

        # Tooltips
        hover_tooltips = [
            ('Year', '@{date}{%Y}'),
            ('Month', '@{date}{%m}'),
            ('Avg. Fare', '$@{avg_fare}{0.2f}'),
            ('Avg. Fare (Low)', '$@{avg_fare_low}{0.2f}'),
            ('Avg. Fare (Lg)', '$@{avg_fare_lg}{0.2f}')
        ]
        
        # Initialize with invisible chart
        if any([self.dropdown_origin.value == '', self.dropdown_destination == '']):
            alpha = 0.
            
        else:
            alpha = 0.8

        # Get filtered data
        filtered_df = self._get_filtered_data()
        filtered_df['date'] = pd.PeriodIndex(
            filtered_df['year'].astype(str) 
            + '-Q' 
            + filtered_df['quarter'].astype(str)
            , freq='Q'
        ).to_timestamp(freq='Q')

        # Aggregate
        avg_fare_by_year = filtered_df.groupby("date").agg(
            avg_fare=("fare", "mean"),
            avg_fare_lg=("fare_lg", "mean"),
            avg_fare_low=("fare_low", "mean")
        ).reset_index()
        
        # Update limits based on values
        self.line_chart_xlim = Range1d(filtered_df['date'].min(), filtered_df['date'].max())
        self.line_chart_ylim = Range1d(
            0, 
            max(
                avg_fare_by_year['avg_fare_lg'].max(), 
                avg_fare_by_year['avg_fare'].max()
            ) + 20
        )

        # Line charts
        avg_fare_line = hv.Curve(
            avg_fare_by_year, 
            "date", 
            "avg_fare", 
            label="Avg Fare"
        ).opts(
            line_color="blue", 
            line_width=2, 
            tools=["hover"], 
            alpha=alpha, 
            hover_tooltips=hover_tooltips
        )

        avg_fare_lg_line = hv.Curve(
            avg_fare_by_year, 
            "date", 
            "avg_fare_lg", 
            label="Avg Fare (Largest Airline)"
        ).opts(
            line_color="#D3D3D3", 
            line_width=2, 
            line_dash="dashed", 
            tools=["hover"], 
            alpha=alpha, 
            hover_tooltips=hover_tooltips
        )

        avg_fare_low_line = hv.Curve(
            avg_fare_by_year, 
            "date", 
            "avg_fare_low", 
            label="Avg Fare (Lowest-Cost Airline)"
        ).opts(
            line_color="#808080", 
            line_width=2, 
            line_dash="dotted", 
            tools=["hover"],
            alpha=alpha, 
            hover_tooltips=hover_tooltips
        )

        # Plot prophet if results present
        if type(self.prophet_fare_df) == pd.DataFrame:

            # Get max historical date
            max_date = filtered_df['date'].max()
            
            # Update limits based on Prophet values
            self.line_chart_xlim = Range1d(filtered_df['date'].min(), self.prophet_fare_df['ds'].max())
            self.line_chart_ylim = Range1d(
                0, 
                max(
                    avg_fare_by_year['avg_fare_lg'].max(), 
                    avg_fare_by_year['avg_fare'].max(), 
                    self.prophet_fare_lg_df['y'].max()
                ) + 20
            )
            
            filtered_prophet_fare_df = (
                self.prophet_fare_df[self.prophet_fare_df.ds >= max_date]
                    .rename(columns={'ds': 'date', 'y': 'avg_fare'})
                    .copy(deep=True)
                    .reset_index(drop=True)
            )

            avg_fare_prophet = hv.Curve(
                filtered_prophet_fare_df, 
                "date", 
                "avg_fare", 
                label="Prophet (Forecast)"
            ).opts(
                line_color="red", 
                line_width=2, 
                tools=["hover"], 
                alpha=alpha, 
                hover_tooltips=hover_tooltips
            )

            filtered_prophet_fare_lg_df = (
                self.prophet_fare_lg_df[self.prophet_fare_lg_df.ds >= max_date]
                    .rename(columns={'ds': 'date', 'y': 'avg_fare_lg'})
                    .copy(deep=True)
                    .reset_index(drop=True)
            )

            avg_fare_lg_prophet = hv.Curve(
                filtered_prophet_fare_lg_df, 
                "date", 
                "avg_fare_lg", 
                label=""
            ).opts(
                line_color="#FF9195", 
                line_width=2, 
                line_dash="dashed", 
                tools=["hover"], 
                alpha=alpha, 
                hover_tooltips=hover_tooltips
            )

            filtered_prophet_fare_low_df = (
                self.prophet_fare_low_df[self.prophet_fare_low_df.ds >= max_date]
                    .rename(columns={'ds': 'date', 'y': 'avg_fare_low'})
                    .copy(deep=True)
                    .reset_index(drop=True)
            )

            avg_fare_low_prophet = hv.Curve(
                filtered_prophet_fare_low_df, 
                "date", 
                "avg_fare_low", 
                label=""
            ).opts(
                line_color="#FF474D", 
                line_width=2, 
                line_dash="dotted", 
                tools=["hover"], 
                alpha=alpha, 
                hover_tooltips=hover_tooltips
            )

            return (avg_fare_line * avg_fare_lg_line * avg_fare_low_line * avg_fare_prophet * avg_fare_lg_prophet * avg_fare_low_prophet).opts(
                xlabel="Date", 
                ylabel="Average Fare",
                title=f"Average Fares Over Time",
                width=800, 
                height=400, 
                legend_position='bottom',
                legend_padding=5,
                legend_spacing=30,
                tools=["hover"],
                margin=self.default_margins
            )

        # Else, plot normal results
        return (avg_fare_line * avg_fare_lg_line * avg_fare_low_line).opts(
            xlabel="Date", 
            ylabel="Average Fare",
            title=f"Average Fares Over Time",
            width=800, 
            height=400, 
            legend_position='bottom',
            legend_padding=5,
            legend_spacing=30,
            tools=["hover"],
            margin=self.default_margins
        )
    

    def _initialize_line_chart(self) -> None:
        """Function to initialize the line charts.
        """
        self.hv_line_chart = hv.DynamicMap(self._redraw_holoviews_line_chart, streams=[UpdateStream()])
        self.bk_line_chart = hv.render(self.hv_line_chart)
        self.bk_line_chart.toolbar.logo = None
        return None


    def _update_line_chart(self) -> None:
        """Updates the line chart when new options are selected.
        """
        self.hv_line_chart.event() # Trigger a redraw, recalculate xlim, ylim
        self.bk_line_chart.x_range = self.line_chart_xlim
        self.bk_line_chart.y_range = self.line_chart_ylim
        return None
    

    # ----------------------------------------------------------------------------------------------
    # Seasonal Boxplot Functions
    # ----------------------------------------------------------------------------------------------
    def _redraw_holoviews_seasonal_boxplot(self) -> hv.Overlay:
        """Function to redefine the seasonal boxplot using the HoloViews API
        """
        
        # Initialize with invisible chart
        if any([self.dropdown_origin.value == '', self.dropdown_destination == '']):
            alpha = 0.
            
        else:
            alpha = 0.8

        # Get filtered data
        filtered_df = self._get_filtered_data()

        # Define the correct season order
        season_order = ['Spring', 'Summer', 'Fall', 'Winter']
        
        # Convert season to categorical with specified order
        filtered_df['season'] = pd.Categorical(
            filtered_df['season'],
            categories=season_order,
            ordered=True
        )
        
        # Sort by season
        filtered_df = filtered_df.sort_values('season')
        
        # Update limits based on values
        self.seasonal_boxplot_ylim = Range1d(0, filtered_df['fare'].max() + 20)

        return hv.BoxWhisker(
            filtered_df,
            'season',
            'fare'
        ).opts(
            title=f"Seasonal Fare Distribution",
            width=800, 
            height=400,
            ylabel="Fare ($)", 
            xlabel="Season",
            box_color='season',
            cmap='Category10',
            show_legend=False,
            box_alpha=alpha,
            box_line_alpha=alpha,
            outlier_alpha=alpha,
            whisker_alpha=alpha,
            margin=self.default_margins
        )
    

    def _initialize_seasonal_boxplot(self) -> None:
        """Function to initialize the seasonal boxplot.
        """
        self.hv_seasonal_boxplot = hv.DynamicMap(self._redraw_holoviews_seasonal_boxplot, streams=[UpdateStream()])
        self.bk_seasonal_boxplot = hv.render(self.hv_seasonal_boxplot)
        self.bk_seasonal_boxplot.toolbar.logo = None
        return None


    def _update_seasonal_boxplot(self) -> None:
        """Updates the seasonal boxplot when new options are selected.
        """
        self.hv_seasonal_boxplot.event() # Trigger a redraw, recalculate ylim
        self.bk_seasonal_boxplot.y_range = self.seasonal_boxplot_ylim
        return None
    

    # ----------------------------------------------------------------------------------------------
    # Airline Chart Functions
    # ----------------------------------------------------------------------------------------------
    def _redraw_holoviews_bar_chart(self, carrier_type: str) -> hv.Bars:
        """Function to redefine the bar chart using the HoloViews API
        """
        
        # Tooltips
        hover_tooltips = [
            ('Avg. Fare', '$@{avg_fare}{0.2f}')
        ]

        # Initialize with invisible chart
        if any([self.dropdown_origin.value == '', self.dropdown_destination == '']):
            alpha = 0.
            
        else:
            alpha = 0.8

        # Get filtered data
        filtered_df = self._get_filtered_data(filter_season=True)

        # Aggregate
        col = 'fare_lg' if carrier_type == 'lg' else 'fare_low'
        name_col = 'carrier_lg_name_concat' if carrier_type == 'lg' else 'carrier_low_name_concat'
        
        avg_data = (
            filtered_df
                .groupby(name_col)
                .agg(avg_fare=(col, "mean"))
                .reset_index()
                .sort_values("avg_fare", ascending=False)[:10]
                .reset_index(drop=True)
        )
        avg_data = avg_data[avg_data['avg_fare'] > 0]

        # Update limits based on values
        self.bar_chart_xlim[carrier_type] = Range1d(0, avg_data['avg_fare'].max() + 20)
        self.bar_chart_ylim[carrier_type] = FactorRange(*avg_data[name_col].to_list())
    
        return hv.Bars(avg_data, name_col, "avg_fare").opts(
            xlabel=f"Average Fare ({'Largest' if carrier_type=='lg' else 'Lowest-Cost'} Carrier)",
            ylabel="Airline",
            title=f"Average Fare ({'Largest' if carrier_type=='lg' else 'Lowest-Cost'} Carrier) by Airline",
            width=800, 
            height=400, 
            tools=["hover"],
            invert_axes=True,
            color="#1f77b4" if carrier_type == 'lg' else "#ff7f0e",
            alpha=alpha,
            hover_tooltips=hover_tooltips,
            margin=self.default_margins
        )


    def _initialize_lg_bar_chart(self) -> None:
        """Function to initialize the "lg" carrier bar chart.
        """
        self.hv_lg_bar_chart = self._redraw_holoviews_bar_chart(carrier_type='lg') # Initial chart
        self.bk_lg_bar_chart = hv.render(self.hv_lg_bar_chart)
        self.bk_lg_bar_chart.toolbar.logo = None
        return None


    def _update_lg_bar_chart(self) -> None:
        """Updates the "lg" carrier bar chart when new options are selected.
        """
        bars = self._redraw_holoviews_bar_chart(carrier_type='lg')
        redrawn_plot = hv.render(bars)
        redrawn_plot.toolbar.logo = None
        self.bk_bar_layout.children[0] = redrawn_plot # force a full redraw to avoid missing elements

        self.bk_lg_bar_chart.x_range = self.bar_chart_xlim['lg']
        self.bk_lg_bar_chart.y_range = self.bar_chart_ylim['lg']
        return None


    def _initialize_low_bar_chart(self) -> None:
        """Function to initialize the "low" carrier bar chart.
        """
        self.hv_low_bar_chart = self._redraw_holoviews_bar_chart(carrier_type='low') # Initial chart
        self.bk_low_bar_chart = hv.render(self.hv_low_bar_chart)
        self.bk_low_bar_chart.toolbar.logo = None
        return None


    def _update_low_bar_chart(self) -> None:
        """Updates the "low" carrier bar chart when new options are selected.
        """
        bars = self._redraw_holoviews_bar_chart(carrier_type='low')
        redrawn_plot = hv.render(bars)
        redrawn_plot.toolbar.logo = None
        self.bk_bar_layout.children[1] = redrawn_plot # force a full redraw to avoid missing elements

        self.bk_low_bar_chart.x_range = self.bar_chart_xlim['low']
        self.bk_low_bar_chart.y_range = self.bar_chart_ylim['low']
        return None
    

    # ----------------------------------------------------------------------------------------------
    # Input Change Callback Functions
    # ----------------------------------------------------------------------------------------------
    def _handle_origin_input_change(self, attr: str, old: str, new: str) -> None:
        """Executed whenever the "Origin" airport changes
        """

        # Update Destination dropdown to relevant values
        if new == '':
            new_options = [''] + sorted(list(
                self.df['airport_name_concat_2'].unique()
            ))
        else:
            new_options = [''] + sorted(list(
                self.df[self.df['airport_name_concat_1'] == new]['airport_name_concat_2'].unique()
            ))

        self.dropdown_destination.options = new_options

        # Update charts
        self._update_choropleth()
        self._update_histogram()
        self.prophet_fare_df = None # remove Prophet results before triggering line chart redraw
        self.prophet_fare_lg_df = None
        self.prophet_fare_low_df = None
        self._update_line_chart()
        self._update_seasonal_boxplot()
        self._update_lg_bar_chart()
        self._update_low_bar_chart()

        return None


    def _handle_destination_input_change(self, attr: str, old: str, new: str) -> None:
        """Executed whenever the "Destination" airport changes
        """

        # Update Origin dropdown to relevant values
        if new == '':
            new_options = [''] + sorted(list(
                self.df['airport_name_concat_1'].unique()
            ))
        else:
            new_options = [''] + sorted(list(
                self.df[self.df['airport_name_concat_2'] == new]['airport_name_concat_1'].unique()
            ))

        self.dropdown_origin.options = new_options

        # Update charts
        self._update_choropleth()
        self._update_histogram()
        self.prophet_fare_df = None # remove Prophet results before triggering line chart redraw
        self.prophet_fare_lg_df = None
        self.prophet_fare_low_df = None
        self._update_line_chart()
        self._update_seasonal_boxplot()
        self._update_lg_bar_chart()
        self._update_low_bar_chart()

        return None


    def _handle_season_input_change(self, attr: str, old: str, new: str) -> None:
        """Executed whenever the "Season" changes
        """

        # Reset prediction
        self._update_analysis_results(0.)

        # Update charts
        self._update_histogram()
        self._update_lg_bar_chart()
        self._update_low_bar_chart()

        return None


    def _handle_ml_model_input_change(self, attr: str, old: str, new: str) -> None:
        """Executed whenever the "ML Model" selection changes
        """

        # Reset prediction
        self._update_analysis_results(0.)

        # Trigger update to line-chart to remove Prophet results
        self.prophet_fare_df = None
        self.prophet_fare_lg_df = None
        self.prophet_fare_low_df = None
        self._update_line_chart()

        return None


    def _handle_analyze_button_click(self, event: ButtonClick) -> None:
        """Executed whenever the "Analyze" button is clicked
        """

        # Get data for inference. Full processed data available in self.df
        origin = self.dropdown_origin.value           # returns "airport_name_concat_1" (SAN - San Diego International Airport)
        destination = self.dropdown_destination.value # returns "airport_name_concat_2"
        season = self.dropdown_season.value           # returns "season"

        # XGBoost / CatBoost
        if self.dropdown_ml_model.value in ('XGBoost', 'CatBoost'):

            # Use the existing function to filter the dataset
            filtered_data = self._get_filtered_data()

            # If no data is found, handle appropriately (e.g., return a default value)
            if filtered_data.empty:
                estimated_price = 0.0

            else:
                # Apply feature engineering to the filtered dataframe
                engineered_data = self.feature_engineering(filtered_data)

                # Select a record (or aggregate as needed) for prediction
                input_data = engineered_data.iloc[[0]]

                # Override the 'year' column with the current year
                input_data.loc[:, 'year'] = datetime.datetime.now().year

                # Ensure the input DataFrame has only the features the model expects
                usable_features = [
                    'year', 'quarter', 'season', 
                    'airport_iata_1', 'airport_iata_2',
                    'state_1', 'state_2',
                    'latitude_1', 'longitude_1',
                    'latitude_2', 'longitude_2',
                    'nsmiles',
                    'lat_diff', 'lon_diff', 'same_state', 'route', 'distance_bin'
                ]
                input_data = input_data[usable_features]

                # Choose model based on dropdown selection
                model_choice = self.dropdown_ml_model.value
                if model_choice == 'XGBoost':
                    estimated_price = self.xgb_model.predict(input_data)[0]
                elif model_choice == 'CatBoost':
                    estimated_price = self.catboost_model.predict(input_data)[0]
                else:
                    estimated_price = 0.0

            # Update analysis results
            self._update_analysis_results(estimated_price)


        # FB Prophet Time Series Forecasting
        elif self.dropdown_ml_model.value == 'FB Prophet':

            # Zero out old predictions, if needed
            self._update_analysis_results(0.)

            # Check for Valid Inputs (note Prophet analysis only needs origin and destination input)
            if (origin == '') or (destination == ''):
                print('Please select valid Origin and Destination from dropdown.')
                return None

            else:

                src, dst = origin.split()[0], destination.split()[0] # get 3 letter code
                route = f'{src}-{dst}'                               # example 'SFO-SAN'
                ts_df = self._get_filtered_data(filter_season=False) # filter to route data ignoring season
                ts_df = ts_df[self.ts_cols]                          # filter to desired cols

                # Check if Data has 'year' ending in 2024
                if ts_df['year'].max() != 2024:
                    print('Selected route ineligible for FB Prophet forecasting.')
                    print('Please select a different route.')
                    return None
                ts_df['date'] = ts_df['year'].astype(str) \
                    .str.cat(ts_df['quarter'].astype(str), sep='-Q')      # example output '2024-Q1'
                ts_df['date'] = pd.PeriodIndex(ts_df['date'], freq='Q') \
                    .to_timestamp(freq='Q')                               # example output '2024-03-31'

                # Check if Data needs to be Aggregated for Codes that have Multiple Airports
                if {src, dst}.intersection(self.multi_airport_codes): # if non-empty set intersection
                    ts_df = ts_df.groupby(['date'], as_index=False, sort=True) \
                        .agg({'fare': 'mean', 'fare_lg': 'mean', 'fare_low': 'mean'})
                    fcst = self.prophet_fare_dfs2.get(route, None) # check if route is valid, returns dict
                    fcst_lg = self.prophet_farelg_dfs2.get(route, None)
                    fcst_low = self.prophet_farelow_dfs2.get(route, None)
                else:
                    ts_df = ts_df.sort_values(by='date').reset_index(drop=True)
                    fcst = self.prophet_fare_dfs.get(route, None)  # check if route is valid, returns dict
                    fcst_lg = self.prophet_farelg_dfs.get(route, None)
                    fcst_low = self.prophet_farelow_dfs.get(route, None)

                # Check if Time Series Data was Eligible for Forecasting (min 50 nonbreaking sequential rows)
                if fcst is None:
                    print('Selected route ineligible for FB Prophet forecasting.')
                    print('Please select a different route.')
                    return None

                # Prophet Forecasts are 8 Qtrs Ahead from 2024-Q1 to 2026-Q1
                fcst_df, fcst_lg_df, fcst_low_df = pd.DataFrame(fcst), pd.DataFrame(fcst_lg), pd.DataFrame(fcst_low)
                fcst_df['ds'], fcst_df['yhat'] = pd.to_datetime(fcst_df['ds']), fcst_df['yhat'].astype(float)
                fcst_lg_df['ds'], fcst_lg_df['yhat'] = pd.to_datetime(fcst_lg_df['ds']), fcst_lg_df['yhat'].astype(float)
                fcst_low_df['ds'], fcst_low_df['yhat'] = pd.to_datetime(fcst_low_df['ds']), fcst_low_df['yhat'].astype(float)

                self.prophet_fare_df = pd.concat(
                    [
                        ts_df[['date', 'fare']].rename(columns={'date': 'ds', 'fare': 'y'}),
                        fcst_df[['ds', 'yhat']].rename(columns={'yhat': 'y'})
                    ],
                    ignore_index=True
                ) # append fcst_df to ts_df
                self.prophet_fare_lg_df = pd.concat(
                    [
                        ts_df[['date', 'fare_lg']].rename(columns={'date': 'ds', 'fare_lg': 'y'}),
                        fcst_lg_df[['ds', 'yhat']].rename(columns={'yhat': 'y'})
                    ],
                    ignore_index=True
                ) # append fcst_df to ts_df
                self.prophet_fare_low_df = pd.concat(
                    [
                        ts_df[['date', 'fare_low']].rename(columns={'date': 'ds', 'fare_low': 'y'}),
                        fcst_low_df[['ds', 'yhat']].rename(columns={'yhat': 'y'})
                    ],
                    ignore_index=True
                ) # append fcst_df to ts_df

                # # Test Output
                # print(
                #     pd.concat(
                #         [
                #             self.prophet_fare_low_df.tail(10).rename(columns={'y': 'fare_low'}),
                #             self.prophet_fare_df[['y']].tail(10).rename(columns={'y': 'fare'}),
                #             self.prophet_fare_lg_df[['y']].tail(10).rename(columns={'y': 'fare_lg'})
                #         ],
                #         axis=1
                #     )
                # )

            # Update analysis charts
            self._update_line_chart()

        # Decision Tree & Random Forest
        elif self.dropdown_ml_model.value in ('Decision Tree', 'Random Forest'):

            # Check for Valid Inputs
            if (origin == '') or (destination == '') or (season == ''):
                print('Please select valid Origin and Destination and Season from dropdown.')
                return None


            # transform user-input into model-appropriate-input
            data = {
                'airport_name_concat_1': [origin],
                'airport_name_concat_2': [destination],
                'season': [season]
            }
            input_df = pd.DataFrame(data)
            # One-hot encode with the same prefix used during training
            input_df_encoded = pd.get_dummies(input_df,
                                              columns=['airport_name_concat_1', 'airport_name_concat_2', 'season'],
                                              prefix=['airport_name_concat_1', 'airport_name_concat_2', 'season'])

            # Dictionary to hold the missing columns with default values of 0
            missing_cols = {col: [0] * len(input_df_encoded) for col in self.decision_forest_model_columns if
                            col not in input_df_encoded.columns}

            # Convert the dictionary to a DataFrame and concatenate it to the existing DataFrame
            missing_df = pd.DataFrame(missing_cols)
            input_df_encoded = pd.concat([input_df_encoded, missing_df], axis=1)

            # Ensure the order of columns matches the training data
            input_df_encoded = input_df_encoded[self.decision_forest_model_columns]


            # Prediction
            model_choice = self.dropdown_ml_model.value
            if model_choice == 'Decision Tree':
                prediction = self.decision_tree_model.predict(input_df_encoded)[0]

            elif model_choice == 'Random Forest':
                prediction = self.random_forest_model.predict(input_df_encoded)[0]
            else:
                prediction = 0.0

            # Update analysis results
            self._update_analysis_results(prediction)

        else:
            print('Error')

        return None


    # ----------------------------------------------------------------------------------------------
    # Webapp Layout
    # ----------------------------------------------------------------------------------------------
    def build(self) -> Union[Row, Column, GridBox, GridPlot]:
        """Builds the webapp layout
        """

        # Build Components
        self._initialize_choropleth()
        self._initialize_histogram()
        self._initialize_line_chart()
        self._initialize_seasonal_boxplot()
        self._initialize_lg_bar_chart()
        self._initialize_low_bar_chart()

        # Inputs & Controls
        controls = [
            self.dropdown_origin,
            self.dropdown_destination,
            self.dropdown_season
        ]
        inputs = row(
            *controls,
            sizing_mode='scale_width', 
            margin=(0, 20, 0, 20),
            styles=Styles(text_align='center', justify_content='center')
        )

        analyze_button = Button(label='Analyze', height=35, width=100, margin=(0, 280, 0, 20), align='end')
        analyze_button.on_click(self._handle_analyze_button_click)
        analyzer_io = row(
            self.dropdown_ml_model, 
            analyze_button,
            self.analysis_results,
            margin=self.default_margins
        )

        # Divs, Titles, and Texts
        title = Div(
            text='<h1>Airfare Price Analyzer</h1>', 
            height=50,
            sizing_mode='stretch_width',
            styles=Styles(text_align='center', justify_content='center'),
            stylesheets=[self.css],
            margin=(0, 20, 20, 20)
        )

        # Generate Layout
        t1 = column(self.bk_histogram, self.bk_seasonal_boxplot)
        t2 = column(self.bk_lg_bar_chart, self.bk_low_bar_chart)

        tabs = Tabs(
            tabs=[
                TabPanel(child=t1, title="Analysis by Fare"),
                TabPanel(child=t2, title="Analysis by Airline")
            ]
        )

        layout = column(
            [
                title,
                inputs,
                row(
                    column(self.choropleth, self.bk_line_chart, analyzer_io), 
                    tabs, 
                    sizing_mode='scale_width'
                )
            ],
            sizing_mode='scale_width'
        )

        self.bk_bar_layout = t2

        return layout