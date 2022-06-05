# Verbal-ConvQuestions

## An Answer Verbalization Dataset for Conversational Question Answering
We introduce a new dataset for conversational question answering over Knowledge Graphs (KGs) with verbalized answers. Question answering over KGs is currently focused on answer generation for single-turn questions (KGQA) or multiple-tun conversational question answering (ConvQA). However, in a real-world scenario (e.g., voice assistants such as Siri, Alexa, and Google Assistant), users prefer verbalized answers. This paper contributes to the state-of-the-art by extending an existing ConvQA dataset with multiple paraphrased verbalized answers. We perform experiments with five sequence-to-sequence models on generating answer responses while maintaining grammatical correctness.
We additionally perform an error analysis that details the rates of models' mispredictions in specified categories. Our proposed dataset (Verbal-ConvQuestions) extended with answer verbalization is publicly **available** with detailed documentation on its usage for wider **utility**.

## Dataset

Similar to [ConvQuestions](https://convex.mpi-inf.mpg.de/) the dataset contains three sets (Train, Val and Test). 

Dataset | Train | Val | Test
--------|-------|-----|-----
Conversations | 6,720 | 2,240 | 2,240
Paraphrased Question | 68,447 | 22,368 | 22,400
Paraphrased Answer | 68,447 | 22,368 | 22,400
Avg. Question length | 8.48 | 8.75 | 8.01
Avg. Answer length | 8.82 | 9.19 | 8.39

The dataset is stored in JSON files and each instance has the following format:

```bash
{
    "conv_id": "Unique conversation id in the dataset",
    "domain": "Domain of the conversation",
    "seed_entity": "Wikidata topic entity of the conversation",
    "seed_entity_text": "Wikidata label of topic entity",
    "questions": [ # list of questions in the conversation
        {
            "question_id": "Unique question id in the conversation",
            "turn": "Actual turn of the question",
            "question": "Question",
            "answer": "Answer of question extracted from Wikidata",
            "answer_text": "Text Answer of question",
            "verbalized_answer": "Initial verbalized answer"
            "paraphrased_answer": ["List of paraphrased answers"],
            "paraphrased_question": ["List of paraphrased questions"],
        },
        ...
    ]
}
```


## Experiments
### Requirements and Setup

Python version >= 3.7

PyTorch version >= 1.8.1

``` bash
# clone the repository
git clone https://github.com/endrikacupaj/Verbal-ConvQuestions.git
cd Verbal-ConvQuestions
pip install -r requirements.txt
```

### Train model
``` bash
# train
python train.py --domain all --model transformer
```

### Test model
``` bash
# test
python test.py --domain all --model transformer --model_path /path/to/checkpoint
```

## License
The dataset is under [Attribution 4.0 International (CC BY 4.0)](dataset/LICENCE)

The software for the experiments is under [MIT License](experiments/LICENCE).

## Cite
Coming Soon!