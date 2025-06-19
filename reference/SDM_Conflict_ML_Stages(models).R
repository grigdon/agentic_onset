
## Replication for Predicting Self-Determination Conflict: A Stage-Specific Framework for Theory Adjudication Using Machine Learning ###


## REQUIRED PACKAGES
library(tidyverse)    
library(dplyr)        
library(haven)      
library(broom)        
library(gt)           
library(caret)        
    

## DATA
onset_escalation_data <- read_csv("/Users/tubit/Desktop/AI Agent Paper/DELPHI_SDM_Conflict_ML_Stages_June14_2025/Data/onset_escalation_data.csv")

data_sdm <- onset_escalation_data %>%
  filter(isrelevant == 1, exclacc == 0)

### Stage 1: Onset Models

# Variables
predictors_stage1 <- c(
   "nviol_sdm_onset",      
   "status_excl",
   "t_claim",
   "lost_autonomy",
   "downgr2_aut",
   "groupsize",
   "lsepkin_adjregbase1",
   "regaut",
   "lgiantoilfield",
   "mounterr",
   "noncontiguous",
   "lnlrgdpcap",
   "lnltotpop",
   "lv2x_polyarchy",
   "lfederal",
   "numb_rel_grps", 
   "coldwar",
   "gwgroupid")


data_sdm_stage1 <- data_sdm %>%
  dplyr::select(all_of(predictors_stage1)) %>%
  drop_na()


data_sdm_stage1$nviol_sdm_onset <- factor(
  data_sdm_stage1$nviol_sdm_onset,
  levels = c(0, 1),
  labels = c("otherwise", "nonviolent_onset")
)

set.seed(666)
unique_groups_1 <- unique(data_sdm_stage1$gwgroupid)
test_groups_1 <- sample(unique_groups_1, size = length(unique_groups_1) * 0.25)
train_data_stage1 <- data_sdm_stage1[!data_sdm_stage1$gwgroupid %in% test_groups_1, ]
test_data_stage2  <- data_sdm_stage1[data_sdm_stage1$gwgroupid %in% test_groups_1, ]

cat("Train Set Unique Groups:", length(unique(train_data_stage1$gwgroupid)), "\n")
cat("Test Set Unique Groups:", length(unique(test_data_stage2$gwgroupid)), "\n")


# drop gwgroupid from train and test data
train_data_stage1$gwgroupid <- NULL
test_data_stage2$gwgroupid <- NULL


set.seed(666)
train_control_1 <- trainControl(
  method ="cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "down",
  savePredictions = "final"
)

## Logistic Regression Models


# Baseline model
set.seed(666)
Base_Onset_Logit <- train(
    nviol_sdm_onset ~
    t_claim+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data      = train_data_stage1,
  method    = "glm",
  metric    = "ROC",
  family="binomial",
  trControl = train_control_1
)

# Complete Model
set.seed(666)
CM_Onset_Logit <- train(
    nviol_sdm_onset ~
    t_claim+
    lv2x_polyarchy+        
    factor(lfederal)+
    factor(regaut)+
    factor(coldwar)+
    factor(status_excl)+
    factor(lost_autonomy)+
    factor(downgr2_aut)+
    factor(lsepkin_adjregbase1) +
    factor(lgiantoilfield)+
    mounterr+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data      = train_data_stage1,
  method    = "glm",
  metric    = "ROC",
  family="binomial",
  trControl = train_control_1
)


# Resource Mobilization
set.seed(666)
RMM_Onset_Logit <- GC_RM_LOG <- train(
    nviol_sdm_onset ~
    t_claim+
    factor(lsepkin_adjregbase1)+
    factor(lgiantoilfield)+
    mounterr+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data      = train_data_stage1,
  method    = "glm",
  metric    = "ROC",
  family="binomial",
  trControl = train_control_1
)


# Political Opportunity
set.seed(666)
POM_Onset_Logit <- train(
    nviol_sdm_onset ~
    t_claim+
    lv2x_polyarchy+
    factor(lfederal)+
    factor(regaut)+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data      = train_data_stage1,
  method    = "glm",
  metric    = "ROC",
  family="binomial",
  trControl = train_control_1
)


# Grievances 
set.seed(666)
GM_Onset_Logit <- train(
    nviol_sdm_onset ~
    t_claim+
    factor(status_excl) +
    factor(downgr2_aut)+
    factor(lost_autonomy)+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data      = train_data_stage1,
  method    = "glm",
  metric    = "ROC",
  family="binomial",
  trControl = train_control_1
)

# Political Opportunity + Grievances
set.seed(666)
POMxGM_Onset_Logit <- train(
    nviol_sdm_onset ~
    t_claim+
    lv2x_polyarchy+
    factor(lfederal)+
    factor(regaut)+
    factor(status_excl) +
    factor(downgr2_aut)+
    factor(lost_autonomy)+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data      = train_data_stage1,
  method    = "glm",
  metric    = "ROC",
  family    ="binomial",
  trControl = train_control_1
)


# Political Opportunity + Resource Mobilization
set.seed(666)
POMxRMM_Onset_Logit <- train(
    nviol_sdm_onset ~
    t_claim+
    lv2x_polyarchy+
    factor(lfederal)+
    factor(regaut)+
    factor(lsepkin_adjregbase1)+
    factor(lgiantoilfield)+
    mounterr+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data      = train_data_stage1,
  method    = "glm",
  metric    = "ROC",
  family="binomial",
  trControl = train_control_1
)


# Resource Mobilization + Grievances
set.seed(666)
RMMxGM_Onset_Logit <- train(
      nviol_sdm_onset ~
      t_claim+
      factor(lsepkin_adjregbase1)+
      factor(lgiantoilfield)+
      mounterr+
      factor(status_excl) +
      factor(downgr2_aut)+
      factor(lost_autonomy)+
      factor(coldwar)+
      factor(noncontiguous) +
      lnlrgdpcap+  
      lnltotpop+             
      groupsize+
      numb_rel_grps,
  data      = train_data_stage1,
  method    = "glm",
  metric    = "ROC",
  family="binomial",
  trControl = train_control_1
)


### Stage 1 Random Forest Models

# Baseline Model
set.seed(666)
Base_Onset_RF <- train(
    nviol_sdm_onset ~
    t_claim+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
  data        = train_data_stage1,
  method      = "rf",
  metric      = "ROC",
  importance  = TRUE,
  ntree       = 1000,
  trControl   = train_control_1
)


# Complete Model
set.seed(666)
CM_Onset_RF <- train(
    nviol_sdm_onset ~
    t_claim+
    lv2x_polyarchy+        
    factor(lfederal)+
    factor(regaut)+
    factor(status_excl)+
    factor(lost_autonomy)+
    factor(downgr2_aut)+
    factor(lsepkin_adjregbase1) +
    factor(lgiantoilfield)+
    mounterr+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data        = train_data_stage1,
  method      = "rf",
  metric      = "ROC",
  importance  = TRUE,
  ntree       = 1000,
  trControl   = train_control_1
)

# Resource Mobilization
set.seed(666)
RMM_Onset_RF <- train(
    nviol_sdm_onset ~
    t_claim+
    factor(lsepkin_adjregbase1)+
    factor(lgiantoilfield)+
    mounterr+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +
    lnltotpop +
    groupsize +
    numb_rel_grps,
  data        = train_data_stage1,
  method      = "rf",
  metric      = "ROC",
  importance  = TRUE,
  ntree       = 1000,
  trControl   = train_control_1
)

# Political Opportunity
set.seed(666)
POM_Onset_RF <- train(
    nviol_sdm_onset ~
    t_claim+
    lv2x_polyarchy+
    factor(lfederal)+
    factor(regaut)+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +
    lnltotpop +
    groupsize +
    numb_rel_grps,
  data        = train_data_stage1,
  method      = "rf",
  metric      = "ROC",
  importance  = TRUE,
  ntree       = 1000,
  trControl   = train_control_1
)

# Grievances
set.seed(666)
GM_Onset_RF <- train(
    nviol_sdm_onset ~
    t_claim+
    factor(status_excl) +
    factor(downgr2_aut)+
    factor(lost_autonomy)+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +
    lnltotpop +
    groupsize +
    numb_rel_grps,
  data        = train_data_stage1,
  method      = "rf",
  metric      = "ROC",
  importance  = TRUE,
  ntree       = 1000,
  trControl   = train_control_1
)

# POM + Grievances
set.seed(666)
POMxGM_Onset_RF <- train(
    nviol_sdm_onset ~
    t_claim+
    lv2x_polyarchy+
    factor(lfederal)+
    factor(regaut)+
    factor(status_excl) +
    factor(downgr2_aut)+
    factor(lost_autonomy)+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data        = train_data_stage1,
  method      = "rf",
  metric      = "ROC",
  importance  = TRUE,
  ntree       = 1000,
  trControl   = train_control_1
)


# POM + RMM
set.seed(666)
POMxRMM_Onset_RF <- train(
    nviol_sdm_onset ~
    t_claim+
    lv2x_polyarchy+
    factor(lfederal)+
    factor(coldwar)+
    factor(regaut)+
    factor(lsepkin_adjregbase1)+
    factor(lgiantoilfield)+
    factor(coldwar)+
    mounterr+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data        = train_data_stage1,
  method      = "rf",
  metric      = "ROC",
  importance  = TRUE,
  ntree       = 1000,
  trControl   = train_control_1
)

# RMM + Grievances
set.seed(666)
RMMxGM_Onset_RF <- train(
    nviol_sdm_onset ~
    t_claim+
    factor(lsepkin_adjregbase1)+
    factor(lgiantoilfield)+
    mounterr+
    factor(status_excl) +
    factor(downgr2_aut)+
    factor(lost_autonomy)+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap+  
    lnltotpop+             
    groupsize+
    numb_rel_grps,
  data        = train_data_stage1,
  method      = "rf",
  metric      = "ROC",
  importance  = TRUE,
  ntree       = 1000,
  trControl   = train_control_1
)







## Stage 2: Violent Escalation Models

firstEscalation_predictors <- c(
  "firstescal",
  "status_excl",
  "lost_autonomy",
  "downgr2_aut",
  "groupsize",
  "lsepkin_adjregbase1",
  "regaut",
  "lgiantoilfield",
  "mounterr",
  "noncontiguous",
  "lnlrgdpcap",
  "lnltotpop",
  "lv2x_polyarchy",
  "lfederal",
  "numb_rel_grps",
  "t_escal",
  "coldwar",
  "gwgroupid"
  )


firstEscalation_data <- data_sdm %>% 
  dplyr::select(all_of(firstEscalation_predictors)) %>%
  drop_na()

firstEscalation_data$firstescal %>% table()

firstEscalation_data$firstescal <- factor(
  firstEscalation_data$firstescal,
  levels = c(0, 1),
  labels = c("otherwise", "first_escalation")
)

set.seed(666) 
unique_groups_fescal_1 <- unique(firstEscalation_data$gwgroupid)
test_groups_fescal_1 <- sample(unique_groups_fescal_1, size = length(unique_groups_fescal_1) * 0.25)

train_data_model_stage2 <- firstEscalation_data[!firstEscalation_data$gwgroupid %in% test_groups_fescal_1, ]
test_data_model_stage2  <- firstEscalation_data[firstEscalation_data$gwgroupid %in% test_groups_fescal_1, ]

cat("Train Set Unique Groups:", length(unique(train_data_model_stage2$gwgroupid)), "\n")
cat("Test Set Unique Groups:", length(unique(test_data_model_stage2$gwgroupid)), "\n")

train_data_model_stage2$gwgroupid <- NULL
test_data_model_stage2$gwgroupid <- NULL



### Logistic Regression Models

# Baseline Model
set.seed(666)
Base_Escalation_Logit <- train(
    firstescal ~ 
    t_escal+
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
  data       = train_data_model_stage2,
  method     = "glm",
  metric     = "ROC",
  family    ="binomial",
  trControl  = train_control_1
)


# Complete Model
set.seed(666)
CM_Escalation_Logit <- train(
        firstescal ~ 
        t_escal +
        lv2x_polyarchy +        
        factor(lfederal) +
        factor(regaut) +
        factor(status_excl) +
        factor(lost_autonomy) +
        factor(downgr2_aut) +
        factor(lsepkin_adjregbase1) +
        factor(lgiantoilfield) +
        mounterr +
        factor(coldwar)+
        factor(noncontiguous) +
        lnlrgdpcap +  
        lnltotpop +             
        groupsize +
        numb_rel_grps,
  data       = train_data_model_stage2,
  method     = "glm",
  metric     = "ROC",
  family=    "binomial",
  trControl  = train_control_1
)

# Resource Mobilization
set.seed(666)
RMM_Escalation_Logit <- train(
    firstescal ~ 
    t_escal +
    factor(lsepkin_adjregbase1) +
    factor(lgiantoilfield) +
    mounterr +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
  data       = train_data_model_stage2,
  method     = "glm",
  metric     = "ROC",
  family     = "binomial",
  trControl  = train_control_1
)

# Political Opportunity
set.seed(666)
POM_Escalation_Logit <- train(
    firstescal ~ 
    t_escal +
    lv2x_polyarchy +
    factor(lfederal) +
    factor(regaut) +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
  data       = train_data_model_stage2,
  method     = "glm",
  metric     = "ROC",
  family     = "binomial",
  trControl  = train_control_1
)


# Grievances
set.seed(666)
GM_Escalation_Logit <- train(
    firstescal ~ 
    t_escal +
    factor(status_excl) +
    factor(lost_autonomy) +
    factor(downgr2_aut) +
    factor(noncontiguous) +
    factor(coldwar)+
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
  data       = train_data_model_stage2,
  method     = "glm",
  metric     = "ROC",
  family     = "binomial",
  trControl  = train_control_1
)

# POM + GM
set.seed(666)
POMxGM_Escalation_Logit <- train(
    firstescal ~ 
    t_escal +
    lv2x_polyarchy +
    factor(lfederal) +
    factor(regaut) +
    factor(status_excl) +
    factor(lost_autonomy) +
    factor(downgr2_aut) +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
  data       = train_data_model_stage2,
  method     = "glm",
  metric     = "ROC",
  family     = "binomial",
  trControl  = train_control_1
)

# POM + RMM
set.seed(666)
POMxRMM_Escalation_Logit <- train(
    firstescal ~ 
    t_escal +
    lv2x_polyarchy +
    factor(lfederal) +
    factor(regaut) +
    factor(lsepkin_adjregbase1) +
    factor(lgiantoilfield) +
    mounterr +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
  data       = train_data_model_stage2,
  method     = "glm",
  metric     = "ROC",
  family     = "binomial",
  trControl  = train_control_1
)


# RMM + GM
set.seed(666)
RMMxGM_Escalation_Logit <- train(
    firstescal ~
    t_escal +
    factor(lsepkin_adjregbase1) +
    factor(lgiantoilfield) +
    mounterr +
    factor(status_excl) +
    factor(lost_autonomy) +
    factor(downgr2_aut) +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
  data       = train_data_model_stage2,
  method     = "glm",
  metric     = "ROC",
  family     = "binomial",
  trControl  = train_control_1
)


### Stage 2:Random Forest Models

# Base Model
set.seed(666)
Base_Escalation_RF <- train(
    firstescal ~ 
    t_escal +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
  data       = train_data_model_stage2,
  method     = "rf",
  metric     = "ROC",
  importance = TRUE,
  ntree      = 1000,
  trControl  = train_control_1
)

# Complete Model
set.seed(666)
CM_Escalation_RF <- train(
  firstescal ~ 
    t_escal +
    lv2x_polyarchy +        
    factor(lfederal) +
    factor(regaut) +
    factor(status_excl) +
    factor(lost_autonomy) +
    factor(downgr2_aut) +
    factor(lsepkin_adjregbase1) +
    factor(lgiantoilfield) +
    mounterr +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
  data       = train_data_model_stage2,
  method     = "rf",
  metric     = "ROC",
  importance = TRUE,
  ntree      = 1000,
  trControl  = train_control_1
)


# Resource Mobilization
set.seed(666)
RMM_Escalation_RF <- train(
    firstescal ~ 
    t_escal +
    factor(lsepkin_adjregbase1) +
    factor(lgiantoilfield) +
    mounterr +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
    data       = train_data_model_stage2,
    method     = "rf",
    metric     = "ROC",
    importance = TRUE,
    ntree      = 1000,
    trControl  = train_control_1
)


# Political Opportunity
set.seed(666)
POM_Escalation_RF <- train(
    firstescal ~ 
    t_escal +
    lv2x_polyarchy +
    factor(lfederal) +
    factor(regaut) +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
    data       = train_data_model_stage2,
    method     = "rf",
    metric     = "ROC",
    importance = TRUE,
    ntree      = 1000,
    trControl  = train_control_1
)

# GM
set.seed(666)
GM_Escalation_RF <- train(
    firstescal ~ 
    t_escal +
    factor(status_excl) +
    factor(lost_autonomy) +
    factor(downgr2_aut) +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
    data       = train_data_model_stage2,
    method     = "rf",
    metric     = "ROC",
    importance = TRUE,
    ntree      = 1000,
    trControl  = train_control_1
)

# POM + GM
set.seed(666)
POMxGM_Escalation_RF <- train(
    firstescal ~ 
    t_escal +
    lv2x_polyarchy +
    factor(lfederal) +
    factor(regaut) +
    factor(status_excl) +
    factor(lost_autonomy) +
    factor(downgr2_aut) +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
    data       = train_data_model_stage2,
    method     = "rf",
    metric     = "ROC",
    importance = TRUE,
    ntree      = 1000,
    trControl  = train_control_1
)

# POM + RMM
set.seed(666)
POMxRMM_Escalation_RF <- train(
    firstescal ~ 
    t_escal +
    lv2x_polyarchy +
    factor(lfederal) +
    factor(regaut) +
    factor(lsepkin_adjregbase1) +
    factor(lgiantoilfield) +
    mounterr +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
    data       = train_data_model_stage2,
    method     = "rf",
    metric     = "ROC",
    importance = TRUE,
    ntree      = 1000,
    trControl  = train_control_1
)

# RMM + GM
set.seed(666)
RMMxGM_Escalation_RF <- train(
    firstescal ~ 
    t_escal +
    factor(lsepkin_adjregbase1) +
    factor(lgiantoilfield) +
    mounterr +
    factor(status_excl) +
    factor(lost_autonomy) +
    factor(downgr2_aut) +
    factor(coldwar)+
    factor(noncontiguous) +
    lnlrgdpcap +  
    lnltotpop +             
    groupsize +
    numb_rel_grps,
    data       = train_data_model_stage2,
    method     = "rf",
    metric     = "ROC",
    importance = TRUE,
    ntree      = 1000,
    trControl  = train_control_1
)






