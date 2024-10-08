import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_cleandata():
    data=pd.read_csv("data/data.csv")
    data=data.drop(['Unnamed: 32','id'],axis=1)
    # print(data.isna().sum())
    data['diagnosis']=data['diagnosis'].map({ 'M':1, 'B':0 })  
    return data


def add_sidbar():
    st.sidebar.title("Cell Nuclei Details")
    data=get_cleandata()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    input_dict={}

    for label,key in slider_labels:
        input_dict[key]=st.sidebar.slider(label,min_value=float(0),max_value=float(data[key].max()),
        value= float(data[key].mean())) 

    return input_dict    

def get_scaled_values(input_dict):
  data = get_cleandata()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

# def get_radar_data(input_data):
#     input_data = get_scaled_values(input_data)
#     categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
#                 'Smoothness', 'Compactness', 
#                 'Concavity', 'Concave Points',
#                 'Symmetry', 'Fractal Dimension']

#     fig = go.Figure()

#     fig.add_trace(go.Scatterpolar(
#         r=[
#           input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
#           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
#           input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
#           input_data['fractal_dimension_mean']
#         ],
#         theta=categories,
#         fill='toself',
#         name='Mean Value '
#     ))

#     fig.add_trace(go.Scatterpolar(
#         r=[
#           input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
#           input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
#           input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
#         ],
#         theta=categories,
#         fill='toself',
#         name='Standard Error'
#     ))

#     fig.add_trace(go.Scatterpolar(
#         r=[
#           input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
#           input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
#           input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
#           input_data['fractal_dimension_worst']
#         ],
#         theta=categories,
#         fill='toself',
#         name='Worst Value'
#     ))

#     fig.update_layout(
#     polar=dict(
#         radialaxis=dict(
#         visible=True,
#         range=[0, 1]
#         )),
#     showlegend=True
#     )

#     return fig

def get_bar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    # Extract values for each category
    mean_values = [
        input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
        input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
        input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
        input_data['fractal_dimension_mean']
    ]

    se_values = [
        input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
        input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
        input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
    ]

    worst_values = [
        input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
        input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
        input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
        input_data['fractal_dimension_worst']
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=mean_values,
        name='Mean Value'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=se_values,
        name='Standard Error'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=worst_values,
        name='Worst Value'
    ))

    fig.update_layout(
        barmode='stack',
        title='Comparison of Mean, Standard Error, and Worst Values',
        xaxis_title='Categories',
        yaxis_title='Values'
    )

    return fig

def add_prediction(input_data):
    # Chemins des fichiers
    model_path = r"model\\random_forest_model.joblib"
    scaler_path = r"model\\scaler.joblib"
    # Charger le modèle et le scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    input_array=np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled=scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
        
    
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Breast Cancer Prediction App",
        page_icon=":hospital:",
        layout="wide",
        initial_sidebar_state="expanded"
    ) 

    input_dict=add_sidbar()
    

    with st.container():
        st.title("Breast Cancer Predictor")
        # Adding a specific instruction with st.write()
        st.write("This app utilizes a machine learning model to predict whether breast cancer diagnoses are benign or malignant based on measurements from your cytology lab. You can update the measurements using the sliders and other functionalities of this app.")
    col1, col2 = st.columns([4,1])  
    with col1:
       figure= get_bar_chart(input_dict)
       st.plotly_chart(figure)
    with col2:
         add_prediction(input_dict)
    
if __name__=='__main__':
    main()