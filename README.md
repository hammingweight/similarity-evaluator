# similarity-evaluator
A Spring AI evaluator that uses cosine similarity to evaluate an LLM response.

## Spring Evaluators
Spring AI has "evaluators" to evaluate the quality of an LLM's response. Two evaluators that are included in Spring AI are the
 * `RelevancyEvaluator`
 * `FactCheckingEvaluator`

 The evaluators use the "LLM as a judge" idiom to assess whether an LLM's response to a prompt is relevant or factually accurate. 
 
Sometimes, though, we have reasonable knowledge about what the LLM's expected answer should be without needing to assess an answer for accuracy and relevance. For example, in evaluating a RAG system, we can write tests that ask
the LLM questions where the documents have a clear answer to the question. The tests can assess whether the LLM's answer is *similar* to an expected answer. We can then optimize our document chunking and retrieval strategies while
ensuring that the correct information is being retrieved from the documents.

Suppose that we've created a chatbot to answer questions about a homeowners' association and the chatbot's knowledge base is augmented with the conduct rules of the association. If we ask the chatbot a question like "How many dogs may be kept in a single home?", we can write a test that the answer is similar to, say, "Residents are limited to owning one dog." Answers like "There is no limit" or "Residents may not possess pets" or "I am unable to find the answer to your question" would be wrong (and dissimilar to a correct answer).

 Given an LLM embedding model, the `SimilarityEvaluator` measures the cosine similarity between an actual and expected answer.

 ## Using the `SimilarityEvaluator`
 ### Configuring an Embedding Model
You need to have configured an embedding model in your `application.properties`. For example,

```
spring.ai.ollama.embedding.enabled=true
spring.ai.ollama.base-url=http://localhost:11434
spring.ai.ollama.embedding.model=qwen3:4b-q4_K_M
```

 ### Using the `SimilarityEvaluator` Class
 Here are snippets of code showing how to use the `SimilarityEvaluator`

```java
import io.github.hammingweight.similarityevaluator.SimilarityEvaluator;

...


String llmAnswer = ...
String goodAnswer = "You may own a single dog";
double minimumSimilarity = 0.9;

SimilarityEvaluator similarityEvaluator = new SimilarityEvaluator(embeddingModel, minimumSimilarity);
EvaluationRequest evaluationRequest = new EvaluationRequest(goodAnswer, llmAnswer);
EvaluationResponse evaluationResponse = similarityEvaluator.evaluate(evaluationRequest);

Assertions.assertTrue(evaluationResponse.isPass());
```

The above code expects that the cosine similarity between the actual and expected answer should be at least 0.9. Sometimes, though, we
don't have a good feeling for what the cosine similarity should be. However, we do know what wrong answers look like. We can then check that the actual answer is more similar to a correct answer than a wrong answer.  For example,

```java
String llmAnswer = ...
String goodAnswer = "You may own a single dog";
String badAnswer = "You may not own a dog";

SimilarityEvaluator similarityEvaluator = new SimilarityEvaluator(embeddingModel);
EvaluationRequest evaluationRequestGood = new EvaluationRequest(goodAnswer, llmAnswer);
EvaluationResponse evaluationResponseGood = similarityEvaluator.evaluate(evaluationRequestGood);
EvaluationRequest evaluationRequestBad = new EvaluationRequest(badAnswer, llmAnswer);
EvaluationResponse evaluationResponseBad = similarityEvaluator.evaluate(evaluationRequestBad);

Assertions.assertTrue(evaluationResponseGood.getScore() > evaluationResponseBad.getScore());
```

## Installing this Package
You can download similarity-evaluator from the [releases page](https://github.com/hammingweight/similarity-evaluator/releases) or install it from the [Github packages Maven repository](https://github.com/hammingweight?tab=packages&repo_name=similarity-evaluator). If you want to use Github's Maven repository, you'll need to have added it to your Maven settings.xml file by following [these instructions](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-apache-maven-registry).

## Building and Testing
### Building the JAR
To build the snapshot JAR

```bash
./gradlew clean jar
ls build/libs/similarity-evaluator-0.0.1-SNAPSHOT.jar
```

If you want to build a specific version, e.g. v0.1.3

```bash
export VERSION=v0.1.3
git checkout $VERSION
./gradlew clean jar
ls build/libs/similarity-evaluator-v0.1.3.jar 
```

### Running the Tests
The integration tests need [ollama](https://ollama.com) and use the `nomic-embed-text:v1.5` embedding model. To run the tests

```bash
ollama pull nomic-embed-text:v1.5
./gradlew clean test
```

