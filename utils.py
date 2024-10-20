import matplotlib.pyplot as plt
import streamlit as st
def bar_chart(rf_probability,xg_boost_probability,knn_probability):
  categories = ['Random Forrest', 'XGBoost','K-Nearest Neighbors']
  values = [float(rf_probability),float(xg_boost_probability),float(knn_probability)]
  print(categories)
  print(values)
  plt.bar(categories,values)
  plt.ylabel('Probability')
  plt.xlabel('Model')

  st.pyplot(plt)