clc
clear
close all

%% -------------------- Dataset -------------------------------------------
% A base de dados corresponde a sinais obtidos através de uma luva
% sensorial que contém 3 acelerômetros. 
% Classes: 
% Possui 2 classes, referentes a dois tipos de movimentos de mão:
% "abrir a mão" (classe -1) {Amostras de 1 a 60} e "mão para baixo" (classe +1) 
% {amostras de 61 a 120}.

% Siniais:
% a matriz de tamanho 1500 x 120. Ou seja, existem 120 objetos de entrada
% (120 movimentos de mão foram realizados), cada um sendo um
% sinal de tamanho 1500.

% Carrega os sinais
load OutputData.mat; load InputData.mat;

rotulos = OutputData;
sinais = InputData';

clear OutputData; clear  InputData;

%% -------------------- Extração de atributos -----------------------------
% Os atributos escolhidos descritos como:

% [1] Média
% [2] Desvio Padrão
% [3] Assimetria "Medida da falta de simetria da distribuição"
% [4] Curtose "caracteriza o achatamento da curva da distribuição"
% [5] entropia
% [6] moda
% [7] Média do modulo da transformada de fourier
% [8] Média da fase da transformada de fourier
% [9] Densidade spectral de potência
% [10] Variança


% Vetor de atributo 
featuresVector = zeros(120,10);

% Nomeia os atributos
nome_features = ["Média","Desvio Padrão","Assimetria","Curtose","Entropia","Moda","Modulo TF","Fase TF","pectral","Variança"];

for i = 1:120 

    % Atributo: Média
    featuresVector(i,1) = mean(sinais(i,:));
    % Atributo: Desvio Padrão
    featuresVector(i,2) = std(sinais(i,:));
    % Atributo: Assimetria
    featuresVector(i,3) = skewness(sinais(i,:));
    % Atributo Curtose
    featuresVector(i,4) = kurtosis(sinais(i,:));
    % Atributo: Entropia
    featuresVector(i,5) = entropy(sinais(i,:));
    % Atributo: Moda
    featuresVector(i,6) = mode(sinais(i,:));
    % Atributo: Média do módulo da transformada de fourier
    featuresVector(i,7) = mean( abs( [fft(sinais(i,1:500)), fft(sinais(i,501:1000)), fft(sinais(i,1001:end))] ) );
    % Atributo: Média da fase da transformada de fourier
    featuresVector(i,8) = mean(angle(fft(sinais(i,:))));
    % Atributo: Densidade spectral de potência
    featuresVector(i,9) = mean(pwelch(sinais(i,:)));
    % Atributo: Variança
    featuresVector(i,10) = var(sinais(i,:));
end



% Normalizando a base dados usando o método F1-score
featuresVector = normalizando_dataset(featuresVector); 

% Plota o grafico de dispersão bidimensional
scatterplot2D(featuresVector,nome_features,rotulos);

% Os atributos que mais são: Média do modulo da transformada de fourier e Densidade spectral de potência
dataset = [featuresVector(:,7), featuresVector(:,9)];


%% -------------------- Treinamento e validação ---------------------------
% A base de dados contendo os atributos é dividida usando o método K-fold. 

Neighbors = 11;
fold = 10;
acuracia_folds = zeros(fold,1);

[ind_teste,ind_treino] = K_fold(dataset,fold);

for i=1:1:fold
    %Atraves dos indices do vetor obtido do método K-fold é selecionado a
    %as amostras de treino e teste com seus respectivos vetores de rotulo
    %da classe pertencente.
    [features_teste, rotulos_teste] = feature_select_rodada(ind_teste(i,:),dataset,rotulos);
    [features_treino, rotulos_treino] = feature_select_rodada(ind_treino(i,:),dataset,rotulos);
    
    %Prediz a que classe pertence os objetos de testes usando o método KNN
    rotulos_preditos = K_Nearest_Neighbors(features_treino,features_teste,rotulos_treino,Neighbors);
    
    %Calcula a taxa de acerto
    acuracia_folds(i,1) = acuracia(rotulos_teste,rotulos_preditos);
end
acuracia_media =  mean(acuracia_folds);
disp('A média da acuracia dos K-folds:');
disp(acuracia_media);




clear features_teste rotulos_teste features_treino rotulos_treino
clear ind_treino ind_teste acuracia_folds
clear rotulos_preditos i featuresVector


%% -------------------- Funções implementadas -----------------------------

% Função que plota os gráficos de dispersão das atributos
function scatterplot2D(dataset,nome_features,rotulos)
    % Inicializa definindo o tamanho do dataset, o número de subplots 
    tam_features = size(dataset); 
    num_subplots = tam_features(2)- 1;

    % calcula os indices que se inicia as amostras das classe
    [~,ind_InicioClasse] = unique(rotulos);
    quan_ind_classes = length(ind_InicioClasse);
    
    
    posicao_subplot = 1;
    soma = 0;
    % Cria uma figura que conterá os scatter plots
    figure
    for i=1:tam_features(2)

        for j=1:tam_features(2)

            if i>j
                % Definine em qual posição se encontra o subplot
                subplot(num_subplots,num_subplots,posicao_subplot+soma)
                
                % Plota as dispersão dos atributos em relação a cada classe
                for k=1:1:quan_ind_classes

                    % Analisa qual o intervalo dos indices de cada classe
                    if k == quan_ind_classes % Se é a ultima classe
                        ind_inicio_classe = ind_InicioClasse(k);
                        ind_fim_classe = tam_features(1,1);
                    else % Caso contrário é pego os 
                        ind_inicio_classe = ind_InicioClasse(k);
                        ind_fim_classe = ind_InicioClasse(k+1)-1;
                    end
                    % Plota a dispersão da classe
                    scatter(dataset(ind_inicio_classe:ind_fim_classe,j),dataset(ind_inicio_classe:ind_fim_classe,i),'filled')
                    hold on
                end

                xlabel(nome_features(j))
                ylabel(nome_features(i))
                hold off
                soma = soma + 1;
            end
           
        end
        if i >= 2
            posicao_subplot = posicao_subplot + num_subplots;
            soma = 0;
        end
    end


end

% Função que calcula a acurácia do modelo
function taxa_acerto = acuracia(rotulos_reais,rotulos_preditas)
    % Tanto rotulos_reais quanto rotulos_preditas são vetores onde os
    % que contem as classes definidas nas linhas e uma unica coluna.
    n_amostras = length(rotulos_reais);
    acertos = 0;

    for i=1:1:n_amostras
        if rotulos_reais(i,1) == rotulos_preditas(i,1)
            acertos = acertos + 1;
        end
    end
    taxa_acerto = (acertos/n_amostras)*100;
    
end

% Função que calcula o modelo de k vizinhos próximos
function classe_feature = K_Nearest_Neighbors(features_treino, features_teste, rotulos_treino, K)
% A função retorna as classes preditas das amostras de teste atravez dos
% da menor distância euclidiana da amostra testada com todas amostras.
% Assim, as k menores amostras visinhas votam para definir em qual classe a
% amostra de treino pertence.

    [n_objetos_testes, ~] = size(features_teste);
    [n_objetos_treino, ~]=size(features_treino);
    distancias = zeros(n_objetos_treino,1);
    classes_visinhos = zeros(K,1);
    classes_selecionada = zeros(n_objetos_testes,1);

    % Prediz qual a classe que pertence a amostra teste por meio da
    % distância euclidiana. 
    for i=1:1:n_objetos_testes
        % Calcula a distância euclidianda da amostra teste com as demais
        % amostras
        distancias(:,1) = distancia_euclidiana(features_teste(i,:),features_treino);
        
        %Pega os indices dos menores distâncias, dessa maneira a função
        %sort organiza o vetor de forma crescente e armazena os indices 
        %dos dados ordenas presevando o indice anterior
        [~, indice_rotulos] = sort(distancias);
        
        % Seleciona os rotulos das menores k distâncias
        for j=1:1:K
            classes_visinhos(j,1) = rotulos_treino(indice_rotulos(j,1),1);
        end
        
        % Retorna a moda das medias, ou seja, pega a classe mais votada
        classes_selecionada(i,1) = mode(classes_visinhos);
    end

    classe_feature =  classes_selecionada;
    
end

% Função que calcula a distância euclidiana entre as amostras
function distancia = distancia_euclidiana(feature_amostra,vetor_features)
% Retorna um vetor com as distâncias euclidianas entre uma amostra teste com as
% demais amostras de treino, assim retornar um vetor (N,1) com as
% distâncias entre um objeto teste com todos os outros objetos treino.

    [n_objetos, ~] = size(vetor_features); 
    vetor_distancia = zeros(n_objetos,1);
    dist_atributos = (feature_amostra-vetor_features).^2;

    for i=1:1:n_objetos
        vetor_distancia(i,1) = sum(dist_atributos(i,:));
    end
    
    distancia = (vetor_distancia).^(1/2);
end

% Função que seleciona os dados de treino e de teste por meio do método k-fold
function [atributos, classes]= feature_select_rodada(indices,features,rotulos)
    % A função dataset_rodada() retorna os atributos da base de dados conforme 
    % os index definidos pelo método k-fold. 
    % A função retorna um vetor contendo os atributos nas colunas e as 
    % amostras nas linhas.

    n = length(indices);
    [~, colunas] = size(features);
    features_selecionados = zeros(n,colunas); %Cria a base de dados dos atributos selecionados
    rotulos_selecionados = zeros(n,1); %Cria a base de dados dos rotulos selecionados
    %Preenche os dados de acordo com a divisão de subgrupos
    for i=1:n
        features_selecionados(i,:) = features(indices(1,i),:);
        rotulos_selecionados(i,1) = rotulos(indices(1,i),1);
    end
    atributos = features_selecionados;
    classes = rotulos_selecionados;
end

% Função que sorteia os indices para treino e teste dos folds do método k-fold
function [teste, treino] = K_fold(data_set,K)
    % O método de cruzamento k-fold consiste em dividir o conjunto total
    % de dados em k subconjuntos mutualmente exclusivos do mesmo tamanho.
    
    % Para implementar o método, a ideia é criar um vetor que indique através 
    % dos indices quem vai ser os objetos de treino e quem vai ser os objetos 
    % de teste, dessa forma possibilitar dividir a base de dados entre
    % treino e teste.
    
    % A função K_fold retorna os indices de treino e os indices de teste.

    [N_objetos,~] = size(data_set); %Descobre quem é a quantidade de atributos 'M' e a quantidade de objetos 'N' 
    dataset_indx = randperm(N_objetos); %cria um vetor aleatório com a permutação 1 atea quantidade de objetos sem repeti-los.
   

    quant_grupo = floor(N_objetos/K); %Define a quantidade de elementos em cada subconjunto K-fold, arredondado 
    vetor_index_teste = zeros(K,quant_grupo);
    vetor_index_treino = zeros(K,N_objetos-quant_grupo);
    ind_inicial = 1; %variavel auxiliar que aponta para o primeiro indice do subconjunto k-fold
    ind_final = quant_grupo; %variavel auxilar que aponta para o ultimo indice do subconjunto k-fold

    for i=1:K

        %Constroi o vetor de indice para os testes
        vetor_index_teste(i,:) = dataset_indx(1,ind_inicial:ind_final);
        
        %Constroi o vetor de indices para os treinos
        if i == 1
            vetor_index_treino(i,:) = dataset_indx(1,(ind_final+1):end);
        elseif i == K
            vetor_index_treino(i,:) = dataset_indx(1,1:(ind_inicial-1));
        else
            vetor_index_treino(i,:) = [dataset_indx(1,1:(ind_inicial-1)), dataset_indx(1,(ind_final+1):end)];
        end
        ind_inicial = ind_final+1; %Indice inicial recebe o ultimo indice mais 1
        ind_final = ind_final+ quant_grupo; %Indice final recebe o final mais a quantidade de elementos do k-fold, já que o periodo se repete em cada k-fold
    end
    teste = vetor_index_teste;
    treino = vetor_index_treino;
end

% Função que normaliza a base de dados usando o método f1-score
function dados = normalizando_dataset(features)
    % Função que normaliza os atributos usando o metodo zscore, assim os
    % atributos devem ter média zero e desvio padrão um.

    [n_amostras, n_atributos] = size(features);

    dados_normalizados = zeros(n_amostras,n_atributos);

    for i=1:1:n_atributos
        dados_normalizados(:,i) = zscore(features(:,i));
    end
    dados = dados_normalizados;
end




