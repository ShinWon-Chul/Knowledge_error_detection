# Incorrect Triple Detection Using Knowledge Graph Embedding and Adaptive Clustering
해당 레파지토리는 [`지식 그래프 임베딩 및 적응형 클러스터링을 활용한 오류 트리플 검출`](/paper/지식그래프%20임베딩%20및%20적응형%20클러스터링을%20활용한%20오류%20트리플%20검출.pdf)의 구현 입니다.  
해당 코드는 지식 그래프를 임베딩 하여 k-means 클러스터링 기반의 오류 검출은 수행합니다.
해당 코드에 대한 직관적인 파악은 [`지식그래프 임베딩 및 적응형 클러스터링을 활용한 오류 트리플 검출`](/paper/지식그래프%20임베딩%20및%20적응형%20클러스터링을%20활용한%20오류%20트리플%20검출.pdf)를 참조 하십시오.

## Library and version
```shell
python 3.6
tensorflow-gpu 1.15.0 - pip install tensorflow-gpu==1.15.0
scikit-learn 0.24.2 - pip install scikit-learn==0.24.2
pandas 1.1.5 - pip install pandas==1.1.5
nltk 3.6.7 - pip install nltk==3.6.7
matplotlib 3.3.4 - pip install matplotlib==3.3.4
seaborn 0.11.2 - pip install seaborn==0.11.2
gensim 3.8.0 - pip install gensim==3.8.0
```

## Data Format
사용한 데이터는 트리플(subject, relation, object)형식을 따릅니다.
DBpedia, FreeBase, WiseKB 세가지 데이터는 실세계의 사람, 도시, 나라, 학교, 스포츠팀, 영화와 같은 개체들과 해당 개체들간의 관계를 나타냅니다. 데이터는 용량관계상 인코딩 되어있는 형식입니다.  

```shell
data/dbpedia/train_positive_20000.txt
e_2398311 r_275 e_651539
e_4856915 r_275 e_651539
e_375534 r_275 e_651539
e_401503 r_275 e_651539
e_1890861 r_275 e_651539
```

- `data/GloVeEntityVectors/glove_*` 폴더는 `*`에 해당하는 데이터의 (entity, type, person) 트리플로부터 생성된 ***Property Sentence***에 대한 [GloVe](https://nlp.stanford.edu/projects/glove/) 알고리즘 임베딩 결과를 나타냅니다.
	- `data/GloVeEntityVectors/glove_dbpedia/person_embedding` : (vocab size, embedding size) shape의 word embedding matrix  
	- `data/GloVeEntityVectors/glove_dbpedia/person_words(list)` : embedding matrix의 행(row)에 해당하는 label word. 
  
- `data/SkipgramEntityVectors/skipgram_*` 폴더는 `*`에 해당하는 데이터의 (entity, type, person) 트리플로부터 생성된 ***Property Sentence***에 대한 [Skip-gram](https://arxiv.org/pdf/1301.3781.pdf) 알고리즘 임베딩 결과를 나타냅니다.  

(***Property Sentence*** : [`지식그래프 임베딩 및 적응형 클러스터링을 활용한 오류 트리플 검출`](/paper/지식그래프%20임베딩%20및%20적응형%20클러스터링을%20활용한%20오류%20트리플%20검출.pdf) 3.1 Property Sentence 생성 참조)

## Running
Skip-gram 임베딩 기반 오류 트리플 검출 실행 파일 : [`Skip-gram_embedding_based_error_triple_detection.ipynb`](/Skip-gram_embedding_based_error_triple_detection.ipynb) 
GloVe 임베딩 기반 오류 트리플 검출 실행 파일 : [`GloVe_embedding_based_error_triple_detection.ipynb`](/GloVe_embedding_based_error_triple_detection.ipynb)

## parameters
- 최적의 오류 검출 결과를 얻기 위한 k-means 클러스터링의 optimal k
	- GloVe
		- DBpedia = 20
		- WiseKB = 34
		- FreeBase = 11
	- Skip-gram
		- DBpedia = 21
		- WiseKB = 27
		- FreeBase = 21

## Experiment Results
|제목|내용|설명|
|------|---|---|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|