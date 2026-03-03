"""Setup script for the credit-card-fraud-detection project."""

from setuptools import setup, find_packages

setup(
    name="credit-card-fraud-detection",
    version="1.0.0",
    description="Credit Card Fraud Detection Using Ensemble Learning and Deep Learning",
    author="Student",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.2.2",
        "xgboost>=1.7.5",
        "imbalanced-learn>=0.10.1",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
    ],
    extras_require={
        "deep_learning": [
            "tensorflow>=2.12.0",
            "torch>=2.0.1",
            "transformers>=4.29.2",
        ],
        "dashboard": ["streamlit>=1.22.0", "plotly>=5.14.1"],
        "explainability": ["shap>=0.41.0", "lime>=0.2.0.1"],
    },
)
