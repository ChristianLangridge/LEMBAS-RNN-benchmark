### Loading in all the data, manipulating the dataframes and generating the right subsets 

result_dir = '/Users/christianlangridge/Desktop/Zhang Lab/Transcriptomics workshop/KallistoResults' # The directory that includes the extracted folders
count_tr = data.frame()
tpm_tr = data.frame()

for(i in list.dirs(result_dir,recursive=F)){
  temp = read.csv(paste0(i,'/abundance.tsv'),sep='\t',stringsAsFactors = F)
  temp_count = data.frame(temp$est_counts)
  temp_tpm = data.frame(temp$tpm)
  colnames(temp_count) = gsub(paste0(result_dir,"/"), "", i)
  colnames(temp_tpm) = gsub(paste0(result_dir,"/"), "", i)
  if(ncol(count_tr) == 0){
    count_tr = temp_count
    rownames(count_tr) = temp$target_id
    tpm_tr = temp_tpm
    rownames(tpm_tr) = temp$target_id
  } else {
    count_tr = cbind(count_tr, temp_count)
    tpm_tr = cbind(tpm_tr,temp_tpm)
  }
}

biomart_file = '/Users/christianlangridge/Desktop/Zhang Lab/Transcriptomics workshop/mart_export.txt' #adjust this based on your file location
mapping = read.csv(biomart_file,sep='\t',stringsAsFactors = F,row.names = 1)
count_gn = merge(count_tr,mapping['Gene.name'], by=0,all=F) # merge only for the shared row names
count_gn = count_gn[,2:ncol(count_gn)]
count_gn = aggregate(.~Gene.name, count_gn, sum)
rownames(count_gn) = count_gn$Gene.name 
tpm_gn = merge(tpm_tr,mapping['Gene.name'], by=0,all=F) # merge only for the shared row names
tpm_gn = tpm_gn[,2:ncol(tpm_gn)]
tpm_gn = aggregate(.~Gene.name, tpm_gn, sum)
rownames(tpm_gn) = tpm_gn$Gene.name

write.table(count_gn,file='/Users/christianlangridge/Desktop/Zhang Lab/Transcriptomics workshop/count_gn.txt',sep = '\t', na = '',row.names = F)
write.table(tpm_gn,file='/Users/christianlangridge/Desktop/Zhang Lab/Transcriptomics workshop/tpm_gn.txt',sep = '\t', na = '',row.names = F)

---------------------------

### Data exploration 
  
metadata_file = '/Users/christianlangridge/Desktop/Zhang Lab/Transcriptomics workshop/metadata.txt' #adjust this based on your file location
metadata = read.csv(metadata_file,sep='\t',stringsAsFactors = F,row.names = 1)

tpm_file = '/Users/christianlangridge/Desktop/Zhang Lab/Transcriptomics workshop/tpm_gn.txt' #adjust this based on your file location
tpm = read.csv(tpm_file,sep='\t',stringsAsFactors = F,row.names = 1)[,rownames(metadata)] # make sure that sequence of metadata is the same with tpm

### PCA analysis
### PCA analysis shows that these samples are very different from one another, each carrying different transcript abundances.
PCA=prcomp(t(tpm), scale=F)
plot(PCA$x,pch = 15,col=c('blue','blue','red','red','lightgreen','lightgreen','black','black'))

---------------------------
  
### Differential gene expression analysis
  
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("DESeq2")
library('DESeq2')

metadata_file = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/metadata.txt' #adjust this based on your file location
metadata = read.csv(metadata_file,sep='\t',stringsAsFactors = F,row.names = 1)
count_file = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/count_gn.txt' #adjust this based on your file location
count = read.csv(count_file,sep='\t',stringsAsFactors = F,row.names = 1)[,rownames(metadata)] # make sure that sequence of metadata is the same with tpm

conds=as.factor(metadata$condition)
coldata <- data.frame(row.names=rownames(metadata),conds)
dds <- DESeqDataSetFromMatrix(countData=round(as.matrix(count)),colData=coldata,design=~conds)
dds <- DESeq(dds)

### Saving comparison between MI_1D and SHAM_1D

cond1 = 'MI_1D' #First Condition
cond2 = 'SHAM_1D' #Reference Condition
res=results(dds,contrast=c('conds',cond1,cond2))
res=data.frame(res)
write.table(res,file='/Users/christianlangridge/Desktop/Zhang Lab/Transcriptomics workshop/deseq_1D.txt',sep = '\t', na = '',row.names = T,col.names=NA)

### Saving comparison between MI_1D and SHAM_3D

cond3 = 'MI_1D' #First Condition
cond4 = 'SHAM_3D' #Reference Condition
res1=results(dds,contrast=c('conds',cond3,cond4))
res1=data.frame(res1)
write.table(res,file='/Users/christianlangridge/Desktop/Zhang Lab/Transcriptomics workshop/deseq_3D.txt',sep = '\t', na = '',row.names = T,col.names=NA)

### Question 1: With Adjusted P-value < 0.05, how many genes are significantly differentially up-regulated? 
### down-regulated? (HINT: Look at Log2FoldChange) 
### What is the most affected gene? (HINT: sort it based on Adjusted P-Value)

### Out of the cohort of genes, no genes are differentialy up-regulated and 3 genes (mt-Co1,mt-Co3 and mt-Nd3) are differentialy down-regulated.
### The most differentially expressed gene is mt-Co3 as it has the lowest padj value.

### Question 2: Why do we have several genes with NA in their statistical result?

### A few genes have NA in their statistical result likely because they were in little abundance that statistical inference couldn't be made
### (either can't determine if it's noise or real signal). 

---------------------------

### Functional analysis
  
deseq_file = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/deseq_1D.txt'
deseq = read.csv(deseq_file,sep='\t',stringsAsFactors = F,row.names = 1)

BiocManager::install("piano")
BiocManager::install("qusage")
BiocManager::install("org.Mm.eg.db")
BiocManager::install("AnnotationDbi")

library('piano')
library('Biobase')
library('snow')
library('RColorBrewer')
library('gplots')
library('visNetwork')
library('org.Mm.eg.db')
library('AnnotationDbi')

GSC='/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/KEGG.gmt'
y=loadGSC(GSC)

input_file=deseq[ ,c('log2FoldChange','pvalue')]
logFC=as.matrix(input_file[,1])
pval=as.matrix(input_file[,2])
rownames(logFC)=toupper(rownames(input_file))
rownames(pval)=toupper(rownames(input_file))
logFC[is.na(logFC)] <- 0
pval[is.na(pval)] <- 1
gsaRes <- runGSA(pval,logFC,gsc=y, geneSetStat="reporter", signifMethod="nullDist", nPerm=1000)

### Looking at why the summary isn't being created.

intersect(rownames(deseq),unlist(y$gsc))
(head(rownames(deseq)))
(head(unlist(y$gsc)))

### Converting over deseq gene names (row names) into ENSEMBL IDs

gene_symbols <- rownames(deseq)

ensembl_ids <- mapIds(org.Mm.eg.db,
                      keys = gene_symbols,     # gene_symbols is your vector of symbols
                      column = "ENSEMBL",
                      keytype = "SYMBOL",
                      multiVals = "first")

rownames(deseq) <- ensembl_ids   

### Same for y$gsc gene names 
valid_symbols <- symbols[symbols %in% keys(org.Mm.eg.db, keytype = "SYMBOL")]

convert_symbols_to_ensembl <- function(symbols) {
  symbols <- symbols[symbols %in% keys(org.Mm.eg.db, keytype = "SYMBOL")]
  mapIds(org.Mm.eg.db, keys = symbols, column = "ENSEMBL", keytype = "SYMBOL", multiVals = "first")
}
gsc_ensembl <- lapply(y$gsc, convert_symbols_to_ensembl)


# Get all valid symbols from org.Mm.eg.db
valid_keys <- keys(org.Mm.eg.db, keytype = "SYMBOL")

# Function to filter each character vector
filter_valid_symbols <- function(symbols) {
  symbols[symbols %in% valid_keys]
}

# Apply over the gene set list
filtered_gsc <- lapply(y$gsc, filter_valid_symbols)



### Rechecking intersection of rownames

intersect(rownames(deseq), unlist(gsc_ensembl)) 
head(unlist(y$gsc))

