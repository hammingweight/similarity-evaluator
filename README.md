# similarity-evaluator
A Spring AI evaluator that uses cosine similarity to evaluate an LLM response.

## Spring Evaluators
Spring AI has "evaluators" to evaluate the quality of an LLM's response. Two evaluators that are included in Spring AI are the
 * `RelevancyEvaluator`
 * `FactCheckingEvaluator`

 The above two evaluators use the "LLM as a judge" idiom to assess whether an LLM's response to a prompt is relevant or factually accurate. 
 
 Sometimes though, we have reasonable knowledge about what the LLM's expected answer should like. For example, in evaluating a RAG system, we can write tests that ask
 the LLM questions where the documents have a clear answer to the question. We can then write tests that simply assess whether the LLM's answer is *similar* to an expected answer. The tests can then assess whether the document chunking and retrieval strategies are working.

 Suppose that we've created a chatbot to answer questions about a homeowners' association and the chatbot's knowledge base is augmented with the conduct rules of the association. If we ask the chatbot a question like "How many dogs may be kept in a single home?", we can write a test that the answer is similar to, say, "Residents are limited to owning one dog." Answers like "There is no limit" or "Residents may not possess pets" or "I am unable to find the answer to your question" would be wrong.

 Given an LLM embedding model, the `SimilarityEvaluator` measures the cosine similarity between an actual and expected answer.

 ## Using the `SimilarityEvaluator`

 Here are snippets of code showing how to use the `SimilarityEvaluator`

```java
String llmAnswer = ...
String goodAnswer = "You may own a single dog";
double minimumSimilarity = 0.9;

SimilarityEvaluator similarityEvaluator = new SimilarityEvaluator(embeddingModel, minimumSimilarity);
EvaluationRequest evaluationRequest = new EvaluationRequest(goodAnswer, llmAnswer);
EvaluationResponse evaluationResponse = similarityEvaluator.evaluate(evaluationRequest);

Assertions.assertTrue(evaluationResponse.isPass());
```

The above good expects that the cosine similarity between the actual and expected answer should be at least 0.9. Sometimes, though, we
don't have a good feeling for what the cosine similarity should be, but we do know what wrongs answers look like. We can then check that the actual answer is more similar to a correct answer than a wrong answer.

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

You need to have configured an embedding model in your `application.properties`. For example,

```
spring.ai.ollama.embedding.enabled=true
spring.ai.ollama.base-url=http://localhost:11434
spring.ai.ollama.embedding.model=qwen3:4b-q4_K_M
```

## Installing this Package
You can download the JAR from the 


