#R
#TCGA
library("rjson")
clinical_tab <-fromJSON("clinical.project-TCGA-BRCA.2017-01-27T11-31-11.841870.json")

get_clinical_data_fromjson<-function(case){
    sample<-substr(as.character(case$demographic$submitter_id),1,12)
    age_at_diagnosis<-as.character(case$diagnoses[[1]]$age_at_diagnosis)
    days_to_last_followup<-as.character(case$diagnoses[[1]]$days_to_last_follow_up)
    days_to_last_disease_status<-as.character(case$diagnoses[[1]]$days_to_last_known_disease_status)
    days_to_death<-as.character(case$demographic$days_to_death)
    
    
    tumour_stage<-as.character(case$diagnoses[[1]]$tumor_stage) #not reported
    vital_status<-as.character(case$demographic$vital_status)
    clin_dat<-c(CaseID=ifelse(is.null(sample),NA,sample),age_at_diagnosis=ifelse(is.null(age_at_diagnosis),NA,age_at_diagnosis),stage=ifelse(is.null(tumour_stage),NA,tumour_stage), vital_status=ifelse(is.null(vital_status),NA,vital_status), days_to_last_followup=ifelse(is.null(days_to_last_followup),NA,days_to_last_followup),
    days_to_last_disease_status=ifelse(is.null(days_to_last_disease_status),NA,days_to_last_disease_status),
    days_to_death=ifelse(is.null(days_to_death),NA,days_to_death))
    return(clin_dat)
}

df<-data.frame(stringsAsFactors=FALSE)

for (i in 1:length(clinical_df)){
    newdf<-get_clinical_data_fromjson(clinical_tab[[i]])
    df<-rbind(df,t(newdf))
}
df$vital_status<-as.character(df$vital_status)
df$days_to_death<-as.character(df$days_to_death)
df$days_to_last_followup<-as.character(df$days_to_last_followup)

df$survival_time<-rep("NA",dim(df)[1])

df[df$vital_status=="Dead","survival_time"]<-df[df$vital_status=="Dead","days_to_death"]
df[df$vital_status=="Alive","survival_time"]<-df[df$vital_status=="Alive","days_to_last_followup"]

df_keep<-df[,c("CaseID","age_at_diagnosis","vital_status","survival_time")]

meta<-read.table("gdc_sample_sheet.2020-06-30.tsv",header=T,sep="\t")
meta<-meta[,c(2,6)]
meta$RNAID<-sapply(as.character(strsplit(meta[,1],'\\.')),function(x)x[[1]])
meta<-meta[,2:3]
colnames(meta)[1]<-"CaseID"
df_ids<-merge(df_keep,meta,by="CaseID")
write.table(df_ids,file="TCGA_OV_RNAseq_clinical.tsv",sep="\t",row.names=F,quote=F)


#AOCS
sampleinfo<-read.table("SampleInformation_withBRCAstatus_genomiconly.txt",sep="\t",header=T)
aocs<-sampleinfo[sampleinfo$Cohort=="AO",]
aocs<-aocs[,c("Sample","WGD","Cellularity","HRDetect","Mutational_load","CNV_load","SV_load")]
surv<-read.table("../Supplementary_tables/HGSOC_clinical_data.txt",sep="\t",header=T)
surv<-surv[surv$cohort=="AOCS",]

aocs_meta<-merge(aocs,surv,by="Sample")
write.table(aocs_meta,file="AOCS_OV_RNAseq_clinical_plusmeta.tsv",sep="\t",row.names=F,quote=F)
