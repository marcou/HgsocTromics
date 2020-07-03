
#You need to load these libraries
#You might need to install them first - which you can do easily from bioconductor in R using the following lines

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# BiocManager::install("biomaRt")
# BiocManager::install("clusterProfiler")
# BiocManager::install("org.Hs.eg.db")

library("clusterProfiler")
library("org.Hs.eg.db")
library("biomaRt")
library(enrichplot)

getKeyGenes<-function(full,IC){
    
    col<-full[,IC]
    three_sds<-3*sd(col)
    av<-mean(col)

    above_ind<-which(col > av+three_sds)
    if (length(above_ind)>0){
        above<-rownames(full)[above_ind]
        df.a<-data.frame(Genes=above,Direction="above",IC=colnames(full)[IC])
    }else{
        df.a<-data.frame()
    }
    
    below_ind<-which(col < av-three_sds)
    if (length(below_ind)>0){
        below<-rownames(full)[below_ind]
        df.b<-data.frame(Genes=below,Direction="below",IC=colnames(full)[IC])
    }else{
        df.b<-data.frame()
    }
    
    
    keygenes<-rbind(df.a,df.b)

    return(keygenes)

}

runGoEnrichments<-function(ICs){
    
    all_ensembl<-rownames(ICs)
        
    all_keygenes<-list()

    for (i in 1:dim(ICs)[2]){
    
        all_keygenes[[i]]<-getKeyGenes(ICs,i)
    
    }
      
    gores<-list()

    for (k in 1:dim(ICs)[2]){
        ego <- enrichGO(gene          = as.character(all_keygenes[[k]][is.na(all_keygenes[[k]][,1])==FALSE,1]),
                        universe      = all_ensembl,
                        OrgDb         = org.Hs.eg.db,
                        keyType       ='ENSEMBL',
                        ont           = "BP",
                        pAdjustMethod = "BH",
                        pvalueCutoff  = 0.05,
                        qvalueCutoff  = 0.05)

        gores[[k]]<-ego
    }
        
    with_enrich_ics<-which(sapply(gores,function(x) dim(head(x))[1] > 0))
    list_names<-sapply(with_enrich_ics,function(y) paste("IC",y,sep=""))
    with_enrich<-gores[with_enrich_ics]
    names(with_enrich)<-list_names
    return(with_enrich)
}

ensembl<-useMart("ensembl",dataset="hsapiens_gene_ensembl")
                       
runKEGGEnrichments<-function(ICs){
    
    all_ensembl<-rownames(ICs)
    gene_anno<-getBM(attributes=c('ensembl_gene_id', 'entrezgene_id'),filters='ensembl_gene_id',values=all_ensembl,mart=ensembl)
    hugo_genes_all<-as.character(gene_anno[gene_anno[,1] %in% all_ensembl,2])
    
    
    all_keygenes<-list()

    for (i in 1:dim(ICs)[2]){
    
        all_keygenes[[i]]<-getKeyGenes(ICs,i)
    
    }
      
    gores<-list()
    
    for (k in 1:dim(ICs)[2]){
        
        genes<-as.character(all_keygenes[[k]][is.na(all_keygenes[[k]][,1])==FALSE,1])
        gene_anno<-getBM(attributes=c('ensembl_gene_id', 'entrezgene_id'),filters='ensembl_gene_id',values=genes,mart=ensembl)
        hugo_genes<-unique(as.character(gene_anno[gene_anno[,1] %in% genes,2]))
        
        ego <- enrichKEGG(gene          = hugo_genes,
                        universe      = hugo_genes_all,
                        organism         = 'hsa',
                        pAdjustMethod = "BH",
                        pvalueCutoff  = 0.05,
                        qvalueCutoff  = 0.05)

        gores[[k]]<-ego
    }
        
    with_enrich_ics<-which(sapply(gores,function(x) dim(head(x))[1] > 0))
    list_names<-sapply(with_enrich_ics,function(y) paste("IC",y,sep=""))
    with_enrich<-gores[with_enrich_ics]
    names(with_enrich)<-list_names
    return(with_enrich)
                       
}

#Run GO and KEGG enrichments with the following commands
#Load your metagene matrix with genes as rows and genenames as rownames, ICs as columns with column names "IC1","IC2"... or something similar
ICs_bygenes<-read.table("/home/ipoole/Documents/gitrepos/HgsocTromics/Factors/TCGA_OV_VST/ICA_median_factor_3.tsv",header=T,row.names=1,sep="\t")

all_go<-runGoEnrichments(ICs_bygenes)
all_kegg<-runKEGGEnrichments(ICs_bygenes)


