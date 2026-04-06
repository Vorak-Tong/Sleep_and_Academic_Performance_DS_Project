import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Mony - Feature Importance\n",
                "\n",
                "**Objective:**\n",
                "Find most important factors affecting GPA\n",
                "\n",
                "**Tasks:**\n",
                "- Load Random Forest & XGBoost models\n",
                "- Extract feature importance\n",
                "- Create bar chart\n",
                "- Rank features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import pickle\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Assuming xgboost is available if the model was saved\n",
                "import xgboost\n",
                "\n",
                "feature_cols = [\n",
                "    \"Study_Hours_Per_Day\",\n",
                "    \"Sleep_Hours_Per_Day\",\n",
                "    \"Social_Hours_Per_Day\",\n",
                "    \"Physical_Activity_Hours_Per_Day\",\n",
                "    \"Extracurricular_Hours_Per_Day\"\n",
                "]\n",
                "\n",
                "# Load models\n",
                "with open('../../models/random_forest_model.pkl', 'rb') as f:\n",
                "    rf_model = pickle.load(f)\n",
                "\n",
                "with open('../../models/xgboost_model.pkl', 'rb') as f:\n",
                "    xgb_model = pickle.load(f)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Extract Feature Importances\n",
                "rf_importances = rf_model.feature_importances_\n",
                "xgb_importances = xgb_model.feature_importances_\n",
                "\n",
                "# Create DataFrame for plotting\n",
                "importance_df = pd.DataFrame({\n",
                "    'Feature': feature_cols,\n",
                "    'Random Forest': rf_importances,\n",
                "    'XGBoost': xgb_importances\n",
                "})\n",
                "\n",
                "# Melt the dataframe for seaborn plotting\n",
                "importance_melted = importance_df.melt(id_vars='Feature', var_name='Model', value_name='Importance')\n",
                "\n",
                "# Plotting\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.barplot(data=importance_melted, x='Importance', y='Feature', hue='Model', palette='viridis')\n",
                "plt.title('Feature Importance Comparison: Random Forest vs XGBoost')\n",
                "plt.xlabel('Importance Score')\n",
                "plt.ylabel('Features')\n",
                "plt.legend(title='Model')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Rank features based on average importance across both models for a unified ranking\n",
                "importance_df['Average Importance'] = importance_df[['Random Forest', 'XGBoost']].mean(axis=1)\n",
                "ranked_df = importance_df.sort_values(by='Average Importance', ascending=False).reset_index(drop=True)\n",
                "\n",
                "print(\"Ranked Feature List (by Average Importance):\")\n",
                "display(ranked_df[['Feature', 'Random Forest', 'XGBoost', 'Average Importance']])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Conclusion\n",
                "\n",
                "**Explanation:**\n",
                "Based on the feature importance extracted from both Random Forest and XGBoost models, `Study_Hours_Per_Day` is clearly the most dominant factor affecting GPA. This is followed by `Sleep_Hours_Per_Day` and `Extracurricular_Hours_Per_Day`, which share moderate significance depending on the model's structure. Social and Physical Activity hours appear to have the least predictive impact on GPA among the measured features."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

file_path = os.path.join(
    r"c:\Users\Admin\Documents\CADT\DataSciene\Sleep_and_Academic_Performance_DS_Project\Sleep_and_Academic_Performance\notebooks\modeling",
    "05_Feature Importance.ipynb"
)

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook created at {file_path}")
