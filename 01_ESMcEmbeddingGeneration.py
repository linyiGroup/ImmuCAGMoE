import sys 
sys.path.append("./") 

from libs.utils.utils import load_config
from libs.utils.ESMcEmbedding import loadEmbeddingDict, \
    fetchUnembeddedDataPathList, fetchUnembeddedSequenceSet, \
    loadESMcModel, saveEmbedding, saveEmbeddingDict



CONFIG_PATH = "./configs/ESMcEmbedding.yaml"

# load config yaml
config = load_config(CONFIG_PATH)

# load embedding dict, if not exist, then return an empty dict
embeddingDict = loadEmbeddingDict(config["embeddingDictPath"])

# fetch the path for each file containing the aa sequences needing to be embedded
dataPathList = fetchUnembeddedDataPathList(config['originDataDirs'])

# get a set, containing all aa sequences of TCR, peptide and MHC pseudo sequence that need to be embedded.
seqSet = fetchUnembeddedSequenceSet(dataPathList, config["squenceColumnName"])

print("total number of seqs that need to be embedded:", len(seqSet))

# load ESMc model
ESMcModel = loadESMcModel(config["ESMcPath"],config["device"])

# embed all seqs, saving the embeddings
embeddingDict = saveEmbedding(ESMcModel, seqSet, embeddingDict, 
                              config["maxSeqLength"], config["embeddingDirPath"], 
                              config["maxFilesPerDir"], config["device"], 
                              config["errorLogPath"])

# save the embedding dict
saveEmbeddingDict(embeddingDict, config["embeddingDictPath"])
print("embedding dict has updated")

print("total number of seqs has been embedded:", len(embeddingDict))