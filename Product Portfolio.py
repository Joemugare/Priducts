#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[42]:


# Load your datasets
df1 = pd.read_csv('Google Products.csv')
df2 = pd.read_csv('Apple Products.csv')
df3 = pd.read_csv('Amazon Products.csv')


# In[43]:


# Concatenate the dataframes vertically
merged_df = pd.concat([df1, df2, df3], ignore_index=True)


# In[53]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_evaluate_model(df):
    # Assuming 'Product Name' and 'Launch Year' are relevant features
    X = df[['Product Name']]
    y = df['Status']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    numeric_features = []  # Add numeric features if any
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Product Name']  # Add categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the classifier
    classifier = RandomForestClassifier(random_state=42)

    # Create and train the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Train the classifier
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Load datasets
df1 = pd.read_csv('Google Products.csv')
df2 = pd.read_csv('Apple Products.csv')
df3 = pd.read_csv('Amazon Products.csv')

# Train and evaluate the model for each dataset
accuracy_google = train_evaluate_model(df1)
accuracy_apple = train_evaluate_model(df2)
accuracy_amazon = train_evaluate_model(df3)

print(f"Accuracy for Google Products: {accuracy_google}")
print(f"Accuracy for Apple Products: {accuracy_apple}")
print(f"Accuracy for Amazon Products: {accuracy_amazon}")


# In[57]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load your datasets
df_google = pd.read_csv('Google Products.csv')
df_apple = pd.read_csv('Apple Products.csv')
df_amazon = pd.read_csv('Amazon Products.csv')

# Add a column to identify the company
df_google['Company'] = 'Google'
df_apple['Company'] = 'Apple'
df_amazon['Company'] = 'Amazon'

# Concatenate the dataframes vertically
df = pd.concat([df_google, df_apple, df_amazon], ignore_index=True)

# Initialize the Dash app
app = dash.Dash(__name__)

# Responsive Design and Styling
app.layout = html.Div(
    style={'width': '80%', 'margin': 'auto', 'font-family': 'Arial, sans-serif'},
    children=[
        html.H1("Product Portfolio Dashboard", style={'color': 'navy', 'text-align': 'center'}),

        # Dropdown for Company
        html.Label("Select Company:", style={'margin-top': '20px'}),
        dcc.Dropdown(
            id='company-dropdown',
            options=[{'label': company, 'value': company} for company in df['Company'].unique()],
            value=df['Company'].unique()[0],
            style={'width': '50%'}
        ),

        # Launch Trends Chart
        dcc.Graph(
            id='launch-trends',
            figure=px.line(
                df, x='Launch Year', y='Status', color='Product Name',
                title='Launch Trends', hover_name='Product Name',
                labels={'Product Name': 'Product'},
            ),
        ),

        # Success Rates Chart
        dcc.Graph(
            id='success-rates',
            figure=px.pie(
                df, names='Status', title='Success Rates', color='Status',
                color_discrete_map={'Active': 'green', 'Discontinued': 'red'}
            ),
        ),

        # Current Status Table
        dcc.Graph(
            id='current-status',
            figure=px.bar(
                df, x='Product Name', y='Launch Year', color='Status', facet_col='Company',
                title='Current Status', labels={'Product Name': 'Product'},
            ),
        ),
    ]
)

# Callbacks to update charts based on selected company
@app.callback(
    [Output('launch-trends', 'figure'),
     Output('success-rates', 'figure'),
     Output('current-status', 'figure')],
    [Input('company-dropdown', 'value')]
)
def update_charts(selected_company):
    filtered_df = df[df['Company'] == selected_company]

    # Launch Trends Chart
    launch_trends_fig = px.line(
        filtered_df, x='Launch Year', y='Status', color='Product Name',
        title=f'Launch Trends - {selected_company}', hover_name='Product Name',
        labels={'Product Name': 'Product'},
    )

    # Success Rates Chart
    success_rates_fig = px.pie(
        filtered_df, names='Status', title=f'Success Rates - {selected_company}', color='Status',
        color_discrete_map={'Active': 'blue', 'Discontinued': 'red'}
    )

    # Current Status Table
    current_status_fig = px.bar(
        filtered_df, x='Product Name', y='Launch Year', color='Status', facet_col='Company',
        title=f'Current Status - {selected_company}', labels={'Product Name': 'Product'},
    )

    return launch_trends_fig, success_rates_fig, current_status_fig

if __name__ == '__main__':
    app.run_server(debug=True)


# In[58]:





# In[68]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load your datasets
df_google = pd.read_csv('Google Products.csv')
df_apple = pd.read_csv('Apple Products.csv')
df_amazon = pd.read_csv('Amazon Products.csv')

# Add a column to identify the company
df_google['Company'] = 'Google'
df_apple['Company'] = 'Apple'
df_amazon['Company'] = 'Amazon'

# Concatenate the dataframes vertically
df = pd.concat([df_google, df_apple, df_amazon], ignore_index=True)

# Convert 'Launch Year' to numeric format
df['Launch Year'] = pd.to_datetime(df['Launch Year'], errors='coerce').dt.year.fillna(1900).astype(int)

# Unique values for filtering options
categories = df['What it does?'].unique()  # Replace 'What it does?' with the correct column name

# Initialize the Dash app
app = dash.Dash(__name__)

# Responsive Design and Styling
app.layout = html.Div(
    style={'width': '80%', 'margin': 'auto', 'font-family': 'Arial, sans-serif'},
    children=[
        html.H1("Product Portfolio Dashboard", style={'color': 'navy', 'text-align': 'center'}),

        # Dropdowns for filtering
        html.Label("Select Company:", style={'margin-top': '20px'}),
        dcc.Dropdown(
            id='company-dropdown',
            options=[{'label': company, 'value': company} for company in df['Company'].unique()],
            value=df['Company'].unique()[0],
            style={'width': '50%'}
        ),

        html.Label("Select Category:", style={'margin-top': '20px'}),
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': category, 'value': category} for category in categories],
            multi=True,
            style={'width': '50%'}
        ),

        # Interactive Elements (Range Slider)
        html.Label("Select Launch Year Range:", style={'margin-top': '20px'}),
        dcc.RangeSlider(
            id='year-slider',
            min=df['Launch Year'].min(),
            max=df['Launch Year'].max(),
            step=1,
            marks={str(year): str(year) for year in range(df['Launch Year'].min(), df['Launch Year'].max() + 1)},
            value=[df['Launch Year'].min(), df['Launch Year'].max()]
        ),

        # Launch Trends Chart
        dcc.Graph(
            id='launch-trends',
        ),

        # Success Rates Chart
        dcc.Graph(
            id='success-rates',
        ),

        # Current Status Table
        dcc.Graph(
            id='current-status',
        ),
    ]
)

# Callbacks to update charts based on selected filters
@app.callback(
    [Output('launch-trends', 'figure'),
     Output('success-rates', 'figure'),
     Output('current-status', 'figure')],
    [Input('company-dropdown', 'value'),
     Input('category-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_charts(selected_company, selected_categories, selected_years):
    filtered_df = df[df['Company'] == selected_company]

    if selected_categories:
        filtered_df = filtered_df[filtered_df['What it does?'].isin(selected_categories)]  # Replace 'What it does?'

    filtered_df = filtered_df[
        (filtered_df['Launch Year'] >= selected_years[0]) & (filtered_df['Launch Year'] <= selected_years[1])
    ]

    # Launch Trends Chart
    launch_trends_fig = px.line(
        filtered_df, x='Launch Year', y='Status', color='Product Name',
        title=f'Launch Trends - {selected_company}', hover_name='Product Name',
        labels={'Product Name': 'Product'},
    ).update_traces(mode='lines+markers', hovertemplate='%{y}<br>%{hovertext}')

    # Success Rates Chart
    success_rates_fig = px.pie(
        filtered_df, names='Status', title=f'Success Rates - {selected_company}', color='Status',
        color_discrete_map={'Active': 'blue', 'Discontinued': 'red'},
        hover_data=['Company'],
    ).update_traces(hoverinfo='label+percent', textinfo='value+percent')

    # Current Status Table
    current_status_fig = px.bar(
        filtered_df, x='Product Name', y='Launch Year', color='Status',
        title=f'Current Status - {selected_company}', labels={'Product Name': 'Product'},
        hover_name='Status',
    ).update_traces(hoverinfo='y+name')

    return launch_trends_fig, success_rates_fig, current_status_fig

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:





# In[ ]:




