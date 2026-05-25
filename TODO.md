# TODOs

## 1. [_] Setup Entropy  
(python env, skrypt pod sbatch, …) [to każde z nas równolegle] + ustalenie ostateczne co do rozmiaru modelu [pod kątem compute]

## 2. [_] Wszystko co pomocne pod trening i eval

1. [_] C4 + training infrastructure 
    - [_] przerzucenie na entropy, wyczyszczenie (bo to jednak masa info z internetu w tym śmieci i boilerplate -> ew. inna baza mniej wymagająca preprocessingu (?)), tokenizacja
    - [_] napisanie pytorchowych DataLoaderów
    - [_] napisanie pętli treningowej [pewnie ważne żeby dość gęsto mieć punkty kontrolne łapane w pętli] -> uniwersalnie pod dense transformer i MoEUT

2. [_] Evaluation infrastructure
    - [_] setup do raportowania lossu, preplexity -> TensorBoard (?)
    - [_] setup do ustalenia zero-shot benchmarków -> ewaluacja na LAMBADA wytrenowanego modelu
    - [_] skrypt (albo coś hookami) do monitorowania MACów (albo FLOPów) i inference time (pod ACT)

## 3. [_] Trening Baselinów

1. [_] Dense transformer  
    - [_]  setup plus sanity check na minibatchu
    - [_] wrzucenienie jobów na cluster, monitorowanie czy nie ma spików lossu ani nic innego niespodziewanego

2. [_] MoEUT  
    - [_] setup plus sanity check na minibatchu - także czy rzeczywiście działa MoE 
    - [_] wrzucenienie jobów na cluster, monitorowanie czy nie ma spików lossu ani nic innego niespodziewanego - w tym monitorowanie wykorzystania ekspertów

## 4. [_] Porównanie danych

-> czy z obu treningów zgadzają się MAC, czy loss curves są ok, jak z performancem MoEUT względem dense transformera

## 5. Fun part czyli nasz rzeczywisty model...