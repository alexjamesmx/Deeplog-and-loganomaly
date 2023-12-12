Steps to run:

1. Add dataset at ./dataset/{folder_name}/{file_name} with any name. (camel case recommended) ("data" is the name I give to txt files).
2. Set train or predict params in ./config/{model_name} as following:
   is_train: false
   is_predict: true
   You can play around with the other parameters for fine tuning

To consider:

1. Set w, h, and step carefully. If a session doesn't fill up the w, padding token from vocab will take the place, this will avoid errors, but definetily won't as efficient as filling sliding windows with real events, so if w=100, use h=5,10,20,50 or divisible that leaves no residue. if w=30 then h my be 3,5,10,etc.
2. The more layers, hidden units, no_epochs, the more time it takes. Consider first testing with default values.
3. Top k is the number of candidates the model will choose from certain num_classes learnt in vocab. if num_clases=30, choose a topk less than num_classes.
   The greater top-k is, the less anomalies it will label. The lower top-k is, more anomalies it will predict. (this may lead to false positives. Try changning values).
4. The code is meant to run also LogAnomaly model, however, this is not complete yet, so forget bout embeddings for now.
5. Deeplog is composed of 2 phases, log key anomaly detection and parameter anomaly detection. This repository uses log key. meaning detecting abnormal sequences where Eventids or eventTemplates from messages are used. Parameter model detection according to some developers doen't improve that much and is more computationally expensive, however. it is being implemented for testing, not finished yet.
6. You can use any dataset just follow the common columns.
7. in "./testing/\*\*" abnormal predictions will be written.
8. First train a model and then predict it, when predicting, change topk values and see what value works better.
9. If an error occurs, it is possible to be from vocab and input sizes. Delete output folder from that dataset and try training again.
10. output folder will be created automallically.
11. Parameters can be declare from command-line. However I'd recommend using config file.
12. Store log class is only for debugging, see file.

How it works:

1. First, parses text files to json file with n number of logs (for testing)
2. Creates sessions of w (window session simulating lapses of time) logs with specific columns. (make sure eventId is in most data, other fields may be discarded)
3. Creates vocab where eventIds are pointed to random indices so when training, try to use normal system execution files. Most known indices created in vocab will be labeled as normal.
4. Create sliding windows with default step 1, where history size is the sub-window of w. E.g w=50, h=10 -> 5 sequences. [[...],[10,20,30,40,50,60,70,80,90,100],[...]]
   labels means the next eventid. if w=10, h=3, s=1 then [10,20,30,40,50,60,70,80,90,100] = [10,20,30] -> 40, [20,30,40] -> 50, [30.40.50] -> 60,[40,50,60] -> 70, [50.60.70] ->80, [60,70,80] ->90, [70,80,90] -> 100

python main.py --config_file config/loganomaly.yaml
python main.py --config_file config/deeplog.yaml
