count_gn = '/Users/christianlangridge/Desktop/count_gene 16.34.09.txt'
count_gene = read.csv(count_gn,sep='\t',stringsAsFactors = F,row.names = 1)

tpm_gn = '/Users/christianlangridge/Desktop/tpm_gene 16.34.09.txt' 
tpm_gene = read.csv(tpm_gn,sep='\t',stringsAsFactors = F,row.names = 1)

mgi_gene_names = rownames(count_gene)

mgi_gene_names <- gsub('"','',mgi_gene_names)
head(mgi_gene_names)
