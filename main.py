import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from data_processing import process_data, tokenize_data
from nn_model import Seq2Seq

if __name__ == "__main__":

    EOS = '_EOS'
    SOS = 'SOS_'

    if (len(sys.argv) < 2) or (len(sys.argv) > 3):
        print("Usage:")
        print("\tmain.py data_path [pretrained_model_path]")
        sys.exit()

    DATA_PATH = sys.argv[1]
    MODEL_PATH = None
    if len(sys.argv) == 3:
        MODEL_PATH = sys.argv[2]

    data = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['en', 'ru'])

    # Data preprocessing
    # Choose the sentences of word-length less than 14, eliminating only 1% of initial data
    max_sentence_length = 14
    data = process_data(data, max_sentence_length, SOS, EOS)

    tokenizer_en = tokenize_data(data.en, vocab_size=2 ** 15)
    tokenizer_ru = tokenize_data(data.ru, vocab_size=2 ** 15)

    encoder_max_length = max([len(tokenizer_en.encode(sentence)) for sentence in data.en]) + 5
    decoder_max_length = max([len(tokenizer_ru.encode(sentence)) for sentence in data.ru]) + 5

    X_train, X_test, y_train, y_test = train_test_split(np.array(data.en), np.array(data.ru), test_size=0.15,
                                                        random_state=15)

    # Model definition
    emb_size = 300
    lstm_hidden_size = 200
    dropout_rate = 0.2

    batch_size = 128
    epochs = 10

    seq2seq_model = Seq2Seq(tokenizer_en,
                            tokenizer_ru,
                            encoder_max_length,
                            decoder_max_length,
                            emb_size,
                            lstm_hidden_size,
                            dropout_rate,
                            SOS,
                            EOS)

    # Training or model weights acquisition
    if MODEL_PATH:
        seq2seq_model.model.load_weights(MODEL_PATH)
        print("Loaded model from disk")
    else:
        print(f"Model training with {epochs} epochs and batch size={batch_size}")
        seq2seq_model.train(X_train, y_train, X_test, y_test, batch_size, epochs)

        # serialize model to YAML
        model_yaml = seq2seq_model.model.to_yaml()
        with open("model_new.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        seq2seq_model.model.save_weights("model_new.h5")
        print("Saved model to disk")


    # Evaluation
    test_gen = seq2seq_model.generate_batch(X_test, y_test, batch_size=1)
    k = -1

    for i in range(20):
        k += 1
        (input_seq, actual_output), _ = next(test_gen)
        decoded_sentence = seq2seq_model.translate_sentence(input_seq)
        print('English input:\t', X_test[k])
        print('Expected translation:', y_test[k].replace(SOS + ' ', "").replace(' ' + EOS, ""))
        print('Provided translation:', decoded_sentence)
        print()


