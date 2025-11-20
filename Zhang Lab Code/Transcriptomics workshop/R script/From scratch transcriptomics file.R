#Loading in data

result_dir = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Results/KallistoResults2'

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

biomart_file = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Data/mart_export.txt'
mapping = read.csv(biomart_file,sep='\t',stringsAsFactors = F,row.names = 1)
count_gn = merge(count_tr,mapping['Gene.name'], by=0,all=F) # merge only for the shared row names
count_gn = count_gn[,2:ncol(count_gn)]
count_gn = aggregate(.~Gene.name, count_gn, sum)
rownames(count_gn) = count_gn$Gene.name 
tpm_gn = merge(tpm_tr,mapping['Gene.name'], by=0,all=F) # merge only for the shared row names
tpm_gn = tpm_gn[,2:ncol(tpm_gn)]
tpm_gn = aggregate(.~Gene.name, tpm_gn, sum)
rownames(tpm_gn) = tpm_gn$Gene.name

write.table(count_gn,file='/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Results/count_gene.txt',sep = '\t', na = '',row.names = F)
write.table(tpm_gn,file='/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Results/tpm_gene.txt',sep = '\t', na = '',row.names = F)

metadata_file = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Data/metadata.txt'
metadata = read.csv(metadata_file,sep='\t',stringsAsFactors = F,row.names = 1)

tpm_file = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Results/tpm_gene.txt'
tpm = read.csv(tpm_file,sep='\t',stringsAsFactors = F,row.names = 1)[,rownames(metadata)]

#Question: How many protein coding transcripts and genes that we have?
#Answer: We have 14 protein coding transcripts/genes in the count_gn/tpm_gn datasets spanning all Kallisto results

PCA=prcomp(t(tpm), scale=F)
plot(PCA$x,pch = 15,col=c('blue','blue','red','red','lightgreen','lightgreen','black','black'))

#Question: What do you think of the sample separations?
#Answer: 

#Differential Expression Analysis

metadata_file = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Data/metadata.txt'
metadata = read.csv(metadata_file,sep='\t',stringsAsFactors = F,row.names = 1)

count_file = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Results/count_gene.txt'
count = read.csv(count_file,sep='\t',stringsAsFactors = F,row.names = 1)[,rownames(metadata)] # make sure that sequence of metadata is the same with tpm

library('DESeq2')
conds=as.factor(metadata$condition)
coldata <- data.frame(row.names=rownames(metadata),conds)
dds <- DESeqDataSetFromMatrix(countData=round(as.matrix(count)),colData=coldata,design=~conds)
dds <- DESeq(dds)

#Retrieving specific comparison results 

cond1 = 'MI_1D' #First Condition
cond2 = 'SHAM_1D' #Reference Condition
res=results(dds,contrast=c('conds',cond1,cond2))
res=data.frame(res)

cond3 = 'SHAM_3D'#2nd reference condition
res2=results(dds,contrast=c('conds',cond1,cond3))
res2=data.frame(res2)

#Save as a .tsv file for (MI_1D vs. SHAM_1D) and (MI_1D vs. SHAM_3D)

write.table(res,file='/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Results/deseq_1D.txt',sep = '\t', na = '',row.names = T,col.names=NA)
write.table(res,file='/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Results/deseq_3D.txt',sep = '\t', na = '',row.names = T,col.names=NA)

#Comparing 1D results

comparison_1D = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Results/deseq_1D.txt'
results_1D = read.csv(comparison_1D,sep='\t',stringsAsFactors = F,row.names = 1)

#Question 1: With Adjusted P-value < 0.05, how many genes are significantly differentially up-regulated? down-regulated?
#(HINT: Look at Log2FoldChange) What is the most affected gene? (HINT: sort it based on Adjusted P-Value) 
#Answer:  

#Question 2: Why do we have several genes with NA in their statistical result?
#Answer: 

#Functional analysis

deseq_file='/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Results/deseq_1D.txt'
deseq = read.csv(deseq_file,sep='\t',stringsAsFactors = F,row.names = 1)

library('piano')
library('Biobase')
library('snow')
library('RColorBrewer')
library('gplots')
library('visNetwork')

GSC='/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Data/KEGG.gmt'
y=loadGSC(GSC)

input_file=deseq[ ,c('log2FoldChange','pvalue')]
logFC=as.matrix(input_file[,1])
pval=as.matrix(input_file[,2])
rownames(logFC)=toupper(rownames(input_file))
rownames(pval)=toupper(rownames(input_file))
logFC[is.na(logFC)] <- 0
pval[is.na(pval)] <- 1
gsaRes <- runGSA(pval,logFC,gsc=y, geneSetStat="reporter", signifMethod="nullDist", nPerm=1000)

res_piano=GSAsummaryTable(gsaRes)

pdf("heatmap.pdf") 
hm = GSAheatmap(gsaRes, adjusted = T)
dev.off()

pdf('/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Code/Transcriptomics workshop/Results/network_plot.pdf') 
nw = networkPlot(gsaRes, class="distinct", direction="both",significance=0.00005, label="names")
dev.off() 

nw_int = networkPlot2(gsaRes, class="distinct", direction="both", significance=0.00005)
#if you want to show it without saving in R, type "nw_int"
visSave(nw_int, file = "network_plot_interactive.html", background = "white")



