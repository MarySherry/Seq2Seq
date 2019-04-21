import numpy as np
from keras.models import Model, model_from_yaml
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding


class Seq2Seq():

    def __init__(self, tokenizer_en, tokenizer_ru, encoder_max_length, decoder_max_length, emb_size, lstm_hidden_size, dropout_rate, SOS, EOS):
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.SOS = SOS
        self.EOS = EOS

        self.tokenizer_en = tokenizer_en
        self.tokenizer_ru = tokenizer_ru

        self.encoder_num_tokens = tokenizer_en.vocab_size
        self.decoder_num_tokens = tokenizer_ru.vocab_size

        self.emb_size = emb_size
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout_rate = dropout_rate

        self.encoder = self.Encoder(self.encoder_num_tokens,
                                    self.emb_size,
                                    self.lstm_hidden_size,
                                    self.dropout_rate)
        self.decoder = self.Decoder(self.decoder_num_tokens,
                                    self.emb_size,
                                    self.lstm_hidden_size,
                                    self.encoder.encoder_states,
                                    self.dropout_rate)

        self.model = Model([self.encoder.encoder_input, self.decoder.decoder_input], self.decoder.decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

        self.encoder_model = self.encoder.encoder_model()
        self.decoder_model = self.decoder.decoder_model()

    class Encoder():
        def __init__(self, encoder_num_tokens, emb_size, lstm_hidden_size, dropout_rate):
            self.encoder_input = Input(shape=(None,), name="Encoder_input")
            self.encoder_emb = Embedding(encoder_num_tokens, emb_size, mask_zero=True, name="Encoder_embedding")
            self.encoder_lstm = LSTM(lstm_hidden_size, return_state=True, name="Encoder_LSTM", dropout=dropout_rate)

            self._model()

        def _model(self):
            embeddings = self.encoder_emb(self.encoder_input)
            encoder_outputs, state_h, state_c = self.encoder_lstm(embeddings)
            self.encoder_states = [state_h, state_c]

        def encoder_model(self):
            return Model(self.encoder_input, self.encoder_states)


    class Decoder():
        def __init__(self, decoder_num_tokens, emb_size, lstm_hidden_size, encoder_states, dropout_rate):
            self.decoder_input = Input(shape=(None,), name="Decoder_input")
            self.decoder_state_input_h = Input(shape=(lstm_hidden_size,), name="Decoder_initial_h_state")
            self.decoder_state_input_c = Input(shape=(lstm_hidden_size,), name="Decoder_initial_c_state")
            self.decoder_initial_state = [self.decoder_state_input_h, self.decoder_state_input_c]
            self.decoder_emb = Embedding(decoder_num_tokens, emb_size, mask_zero=True, name="Decoder_embedding")
            self.decoder_lstm = LSTM(lstm_hidden_size, return_sequences=True, return_state=True, name="Decoder_LSTM",
                                     dropout=dropout_rate)
            self.decoder_dense = Dense(decoder_num_tokens, activation='softmax')

            self._train_model(encoder_states)
            self._inference_model()

        def _train_model(self, encoder_states):
            embeddings = self.decoder_emb(self.decoder_input)
            decoder_outputs, state_h, state_c = self.decoder_lstm(embeddings, initial_state=encoder_states)
            self.decoder_outputs = self.decoder_dense(decoder_outputs)

        def _inference_model(self):
            embeddings = self.decoder_emb(self.decoder_input)
            decoder_outputs, state_h, state_c = self.decoder_lstm(embeddings, initial_state=self.decoder_initial_state)
            self.decoder_states = [state_h, state_c]
            self.decoder_inference_outputs = self.decoder_dense(decoder_outputs)

        def decoder_model(self):
            return Model([self.decoder_input] + self.decoder_initial_state,
                         [self.decoder_inference_outputs] + self.decoder_states)

    def train(self, X_train, y_train, X_test, y_test, batch_size=128, epochs=10):
        '''Train model given train and validation data'''
        train_samples = len(X_train)
        val_samples = len(X_test)

        self.model.fit_generator(generator=self.generate_batch(X_train, y_train, batch_size=batch_size),
                                 steps_per_epoch=train_samples // batch_size,
                                 epochs=epochs,
                                 validation_data=self.generate_batch(X_test, y_test, batch_size=batch_size),
                                 validation_steps=val_samples // batch_size)

    def generate_batch(self, X, y, batch_size=128):
        ''' Generate a batch of data for encoder, decoder inputs '''
        while True:
            for j in range(0, len(X), batch_size):
                encoder_input_data = np.zeros((batch_size, self.encoder_max_length), dtype='float32')
                decoder_input_data = np.zeros((batch_size, self.decoder_max_length), dtype='float32')
                decoder_target_data = np.zeros((batch_size, self.decoder_max_length, self.decoder_num_tokens), dtype='float32')
                for i, (input_text, target_text) in enumerate(zip(X[j:j + batch_size], y[j:j + batch_size])):
                    encoded_input_sentence = self.tokenizer_en.encode(input_text)
                    encoder_input_data[i, 0:len(encoded_input_sentence)] = encoded_input_sentence

                    encoded_target_sentence = self.tokenizer_ru.encode(target_text)
                    decoder_input_data[i, 0:len(encoded_target_sentence)] = encoded_target_sentence
                    for t, word in enumerate(encoded_target_sentence):
                        if t > 0:
                            decoder_target_data[i, t - 1, word] = 1.
                yield ([encoder_input_data, decoder_input_data], decoder_target_data)

    def decode_sequence(self, input_seq):
        '''Performs a sentence encoding and
        a sequential decoding of encoded vector word by word till EOS symbol'''
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Initialize the translation with Start-of-String character
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.tokenizer_ru.encode(self.SOS)[0]

        decoded_sentence = ''
        while True:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.tokenizer_ru.decode([sampled_token_index])

            # Stop translation upon exceeding of translated elements or achieving of stop character
            if (sampled_char.replace("\\&undsc", "_") == self.EOS or
                    len(decoded_sentence) > 50):
                break
            decoded_sentence += sampled_char
            # Update the target sequence by the current translated element
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]
        return decoded_sentence

    def translate_sentence(self, sentence):
        '''Translate given sentence'''
        decoded_sentence = self.decode_sequence(sentence)
        return decoded_sentence