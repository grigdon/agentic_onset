# Comparing AI-Tuned vs. Human-Tuned Random Forest Models for Conflict Prediction

This project explores how a Large Language Model (LLM) can optimize machine learning models compared to traditional human-guided methods. Specifically, it uses Random Forest models to predict political conflict "onset" (when nonviolent conflict begins) and "escalation" (when nonviolent conflict turns violent).

## How It Works (A High-Level Overview)

At its core, this script sets up a "competition" between two approaches to building a good predictive model:

1.  **The "Human-Tuned" Baseline:** This represents a standard, robust approach to tuning a Random Forest model, similar to how an experienced data analyst might do it using established best practices (like those found in R's `caret` package). It involves careful cross-validation and specific strategies to handle imbalanced datasets.

2.  **The "AI-Tuned" Approach:** Here, an advanced AI (a Large Language Model like GPT-4) is given the task of suggesting the best settings (hyperparameters) for the Random Forest model. The AI iteratively learns from its previous suggestions, aiming to improve the model's performance on a dedicated validation set.

The goal is to see if the AI can discover optimal model settings that lead to better predictive performance than the human-tuned baseline when both are evaluated on completely unseen data.

## Key Steps of the Project

The script follows a structured process for each prediction stage ("onset" and "escalation") and for various theoretical models (different sets of input variables):

1.  **Data Preparation:**
    * The script first loads the raw conflict data.
    * It then applies specific filters, mirroring exact data exclusions from previous research to ensure consistency.
    * Crucially, for each prediction stage (onset/escalation), it cleans the data by removing any rows with missing information in the variables relevant to *any* of the theoretical models for that stage.
    * Finally, the data is split into a main "training" set and a held-out "test" set. This split is "group-aware," meaning that all observations related to a single conflict group (`gwgroupid`) stay together, either in the training or test set. This prevents the model from "cheating" by seeing parts of a conflict it's supposed to predict.

2.  **Human-Tuned Model Training:**
    * For each theoretical model, a Random Forest classifier is trained using parameters and a tuning strategy designed to replicate a well-established "human-tuned" approach from statistical software (R's `caret` package).
    * This involves an internal cross-validation process where the model's `max_features` (a key Random Forest setting) is optimized.
    * To handle class imbalance (where one outcome, like "escalation," is much rarer than "no escalation"), a form of "downsampling" is applied to the training data.

3.  **AI-Tuned Model Optimization:**
    * If enabled, the LLM takes over for this phase. The *overall training set* is further divided into an "AI-training" set and an "AI-validation" set (again, using a group-aware split).
    * The LLM is given information about the model's purpose, its current performance (based on the human baseline), and a history of its own previous suggestions.
    * The LLM then suggests a new set of Random Forest hyperparameters.
    * A Random Forest model is built with these suggested parameters and evaluated on the "AI-validation" set.
    * This feedback (the validation performance) is given back to the LLM, allowing it to refine its suggestions over several iterations.

4.  **Final Model Training and Evaluation:**
    * Once the human-tuned model is optimized, and the AI has found its "best" parameters, both models are then re-trained using their respective optimal settings on the *entire* main training dataset.
    * Both the human-tuned and AI-tuned models are then rigorously evaluated on the completely unseen "test" dataset. This ensures a fair comparison of their true predictive capabilities. The primary metric used for comparison is the Area Under the Receiver Operating Characteristic Curve (AUC), which is particularly useful for imbalanced classification problems.

5.  **Results and Reporting:**
    * The script generates plots (like ROC curves and bar charts) to visually compare the performance of the human-tuned and AI-tuned models.
    * A detailed text report is also produced, summarizing AUC scores, classification reports, feature importances, and the AI's optimization journey for each theoretical model and prediction stage.
    * All raw results and plotting data are saved for further analysis.

## Running the Script

To run this script, you will need:

* **Python 3.x**
* The required Python libraries (listed at the top of the script; you can install them using `pip install pandas numpy scikit-learn matplotlib seaborn tqdm openai`).
* A data file named `onset_escalation_data.csv` in the same directory as the script.
* (Optional, for AI tuning) An [OpenAI API key](https://platform.openai.com/account/api-keys).

You can run the script from your terminal:

```bash
python your_script_name.py --api_key YOUR_OPENAI_API_KEY