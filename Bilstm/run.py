import sys

if len(sys.argv) < 2:
    print("Invalid Option Selected")
elif sys.argv[1] == "train":
    # run train
    import subprocess
    subprocess.call(['python', 'train_test.py', '--train=True', '--cuda=True'])
elif sys.argv[1] == "test":
    # run test
    import subprocess
    subprocess.call(['python', 'train_test.py', '--decode=True'])
elif sys.argv[1] == "vocab":
    # create vocab list
    import subprocess
    subprocess.call(['python', 'vocab.py'])
else:
    print("Invalid Option Selected")