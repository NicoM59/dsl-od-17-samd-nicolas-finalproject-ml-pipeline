import pytest
import pandas as pd
import numpy as np
from app.preprocess import clean_body_text, map_canonical_category
from app.train import create_pipeline

# --- FIXTURES ---
@pytest.fixture
def sample_mental_health_data():
    """Pour les tests de preprocessing et schéma"""
    return pd.DataFrame({
        'body': ["I feel very anxious today", "  Checking text spacing  ", np.nan],
        'category': ["Anxiety", "adhd", "UnknownCategory"]
    })

@pytest.fixture
def mini_training_dataset():
    """Pour le test d'intégration du modèle"""
    X_train = pd.Series([
        "I feel so depressed and sad every day",
        "My anxiety is through the roof, I am shaking",
        "I have racing thoughts and manic episodes due to my bipolar",
        "Voices in my head and hallucinations, classic schizophrenia symptoms",
        "I struggle with focus and hyperfocus because of my adhd",
        "Struggling with social interactions and sensory overload, autism life",
        "Emotional instability and fear of abandonment, typical bpd"
    ])
    y_train = pd.Series([
        "Depression", "Anxiety", "Bipolar", "schizophrenia", "ADHD", "Autism", "BPD"
    ])
    return X_train, y_train

# --- TESTS DE DONNÉES / SCHÉMA ---
def test_validate_dataset_schema(sample_mental_health_data):
    required_columns = ['body', 'category']
    assert all(col in sample_mental_health_data.columns for col in required_columns)

def test_clean_body_text_spaces():
    assert clean_body_text("   Post avec espaces en trop   ") == "Post avec espaces en trop"

def test_handle_nan_values():
    assert map_canonical_category(np.nan) is None
    assert map_canonical_category(None) is None

@pytest.mark.parametrize(
    "input_label, expected_output",
    [
        ("adhd", "ADHD"),
        ("anxiety", "Anxiety"),
        ("bipolar", "Bipolar"),
        ("schizo", "schizophrenia"),
        ("depression", "Depression"),
        ("bpd", "BPD"),
        ("unknown", None),
    ]
)
def test_all_canonical_labels_parametrize(input_label, expected_output):
    assert map_canonical_category(input_label) == expected_output

# --- TEST DU MODÈLE (END-TO-END) ---
@pytest.mark.fast  # Marqueur comme dans ton cours
def test_model_end_to_end_pipeline(mini_training_dataset):
    X_train, y_train = mini_training_dataset
    pipeline = create_pipeline()
    
    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Inférence
    new_post = ["I cannot focus on my work today, my mind wanders"]
    prediction = pipeline.predict(new_post)
    
    # Vérifications
    assert len(prediction) == 1
    assert isinstance(prediction[0], str)
    assert prediction[0] in y_train.values