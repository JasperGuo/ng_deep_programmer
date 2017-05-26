## Deep Learning Model for Ng deep Programmer

- **Basic Model**

  A prototype of the whole idea

- **ff_basic_model**

  Memory Encoder and Output Encoder share the weights. ( **DNN Encoder** )

  *Position Embedding* is added to the word embedding, when encoding value

- **rnn_basic_model**

  Encode memory and output with **LSTM**

- **ff_context_model**

  Memory Encoder and Output Encoder share the weights (**DNN Encoder**)

  Encode local context for each memory entry

---

- **vocab_manager.py**

  Manage vocabulary, `id2word` and `word2id`

  **vocabs** contains digit, lambda, operation, data_type vocabulary

- **models_util.py**

  Contains utility function for model implementation

  `softmax_with_mask`: calculate softmax with a mask, indicating which parts of the tensor is useless

  `get_last_relevant`: retrieve the last hidden states from RNN output

- model/**data_iterator.py**

  `Batch`: A data collection to feed network

  `DataIterator`: Manage Batch

- model/configuration/***.json**

  Network hyperparameters

- model/**model.py**

  Implementation of Network

- model/**checkpoint**/*

  Containing checkpoint of each training process

  - *tf_best_checkpoint* records the parameters with highest dev accuracy in training
  - tf_checkpoint records the parameters of the last epoch

- model/**result_log**/*

  Record each epoch information, (train accuracy, dev accuracy and loss)

- model/**log**/*

  Used for tensorboard

---

### To run the model:

1. Create a directory named **feed_tf** in current directory, and copy the training data to the directory

   For example: `learning_models/feed_tf/step_0_train.json`

2. Go to one the model directory, `learning_models/ff_context_model`

3. Modify hyper parameters in `configuration/conf.json`

4. Run `runtime.py`

   Option:

   | Option       | Description             |
   | ------------ | ----------------------- |
   | --conf       | Configuration file path |
   | --test       | Run test model          |
   | --log        | Log the inference       |
   | --checkpoint | Checkpoint file path    |

```
# Run in training mode
python runtime.py --conf ./configuration/conf.json

# Run in test mode
python runtime.py --conf ./configuration/conf.json --test --log --checkpoint ./checkpoint/tf_best_checkpoint
```