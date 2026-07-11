from pathlib import Path
import sys
from src.crypto_signals.data import make_synthetic_crypto_data
from src.crypto_signals.model import SignalModel

if __name__ == '__main__':
    frame = make_synthetic_crypto_data(n_rows=500)
    dataset = SignalModel.build_training_dataset([frame])
    model = SignalModel(model_path='models/crypto_signal_v1.joblib')
    metrics = model.fit(dataset.drop(columns=['target']), dataset['target'])
    print('metrics=', metrics)
    model.save()
    print('saved=', Path('models/crypto_signal_v1.joblib').exists())
