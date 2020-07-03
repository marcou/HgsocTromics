#Extract files from folders
for i in */;
do
    cd $i
    cp *.counts.gz ../../TCGA_OV_rawcounts/
    cd ..
done

R
#Combine files in R
files<- list.files(pattern = "\\.(counts.gz)$")
filenames<-gsub("[.]htseq.counts.gz","",files)
filenames2<-gsub("-","_",filenames)
filenames3<-paste("T",filenames2,sep="")

for (i in 1:length(files)){
    assign(filenames2[i],read.table(gzfile(files[i]),col.names=c("Gene",filenames2[i])))
    print(i)
}

TCGA_RNA<-get(filenames2[1])

for (j in 1:(length(files)-1)){
    TCGA_RNA<-merge(TCGA_RNA,get(filenames2[j+1]),by= "Gene",all=TRUE)
    print(j)
}

rownames(TCGA_RNA)<-as.character(TCGA_RNA[,1])

TCGA_RNA_keep<-TCGA_RNA[,-1]
genes<-rownames(TCGA_RNA_keep)
genes2<-sapply(strsplit(genes,'\\.'),function(x) x[1])
write.table(genes2,file="~/Desktop/all_genes2.txt",sep="\t",col.names=F,row.names=F,quote=F)

#Use biomart to get gene types
gene_types2<-read.table("../genes_type2.txt",sep="\t",header=T)
prot_coding_genes<-as.character(gene_types2[gene_types2[,2]=="protein_coding",1])
rownames(TCGA_RNA_keep)<-genes2
TCGA_RNA_keep_prot_cod<-TCGA_RNA_keep[prot_coding_genes,]

coldata<-data.frame()
s<-gsub('_','-',colnames(TCGA_RNA_keep_prot_cod))
Sample<-gsub('X','',as.character(s))
coldata<-data.frame(Sample)
colnames(coldata)[1]<-"Sample"
coldata[,1]<-as.character(coldata[,1])

library("DESeq2")
dds<- DESeqDataSetFromMatrix(countData = TCGA_RNA_keep_prot_cod,
                                         colData=coldata,
                                         design=~1)
vsd <- vst(dds, blind=TRUE)

all_vsd<-assay(vsd)
colnames(all_vsd)<-coldata$Sample
write.table(all_vsd,file="TCGA_OV_VST.txt",sep="\t",quote=F)

