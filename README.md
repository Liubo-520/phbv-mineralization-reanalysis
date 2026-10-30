# PHBV mineralization: research code and frozen model

Core numerical code for endpoint-harmonized PHBV mineralization analysis and
the frozen pre-experiment random-forest model (version 3.0.0).

The model was fitted to 1,031 CO2-mineralization observations from 40 curves
and 11 studies. All preprocessing is included in the bundle. Identifiers are
used for grouping and provenance, never as predictive features.

## Run the model

Use Python 3.10 or later and install the pinned inference dependencies:

~~~sh
python -m pip install -r requirements.txt
python predict.py --example
~~~

For your own input, supply a JSON object using the feature names in
models/model_card_metrics.json:

~~~sh
python predict.py --input my_conditions.json
~~~

The explicit example uses the representative conditions stored in the model
bundle. It demonstrates software operation, not a new experiment.

## Reconstruct the public observation record

The original measurements and public workbook/CSV accompany:

Kotzabasaki et al. Machine Learning Methods for Mineralization-Based
Biodegradation Prediction in Polyhydroxyalkanoate-Based Biopolymers:
Insights from Lab-Scale Experiments. Polymers 2026, 18, 1076.
https://doi.org/10.3390/polym18091076

Obtain PHBV_Biodegradation_Curated_Raw_Dataset.xlsx and
biodegradation_loss_data_LM.csv from the publication's public repository:
https://github.com/FSL-AUA/Biodegradation-model

The reconstruction command checks their expected SHA-256 values before use.
Observation tables and source-article PDFs are obtained from those sources;
this compact distribution supplies code, the trained model, and model metadata.

~~~sh
python -m pip install -r requirements-analysis.txt
python reconstruct_data.py --workbook PHBV_Biodegradation_Curated_Raw_Dataset.xlsx --published-csv biodegradation_loss_data_LM.csv --output outputs/reconstructed
python evaluate.py --data outputs/reconstructed/processed/D3_endpoint_audited_mineralization.csv --output outputs/evaluation
~~~

Evaluation defaults to F-pre predictors, random_state=42, nested curve-family
evaluation, and leave-one-study-out evaluation. Learned preprocessing is fitted
inside each calibration fold. The full five-model nested analysis is
computationally intensive. A quick plumbing check can use --models mean and
--bootstrap-iterations 25; that abbreviated check does not replace the full
analysis.

The src directory also retains the numerical sensitivity, kinetic,
uncertainty, and calibration-design routines. Its extended R1 orchestration
expects the corresponding locally assembled audit and intermediate files.
The commands above expose reconstruction and the primary hierarchical model
evaluation directly without manuscript or editorial files.

## Interpretation

The frozen all-data fit is an executable research artifact. It is distinct
from the fold-specific models selected during nested evaluation. Model-card
fields identify the exact frozen configuration and its evaluation summaries.
Cross-study evaluation did not establish a random-forest advantage over a
time-only kinetic reference. The broad cross-study residual bands are
descriptive stress summaries, not prospective coverage guarantees.

Use the artifact for transparent research comparisons. Independent laboratory
validation is needed before material ranking, experimental replacement,
certification, or environmental decision-making.

## Verify this distribution

~~~sh
python -m pip install -r requirements-analysis.txt
python -m pytest -q
~~~

SHA256SUMS and release_manifest.json identify every distributed file. The
joblib bundle and model-card JSON are byte-identical to the versioned research
artifacts. The release contains no observation tables, manuscripts,
correspondence, source PDFs, credentials, or private repository history.
