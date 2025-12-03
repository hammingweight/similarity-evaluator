package io.github.hammingweight.similarityevaluator;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.evaluation.EvaluationRequest;
import org.springframework.ai.evaluation.EvaluationResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
@SpringBootApplication
public class SimilarityEvaluatorIntegrationTests {
	
	@Autowired
	public EmbeddingModel embeddingModel;

	
	@Test
	public void testGoodSimilarity() {
		SimilarityEvaluator evaluator = new SimilarityEvaluator(embeddingModel);
		EvaluationRequest request = new EvaluationRequest("the cat sat on the mat", "the cat sat on the mat");
		EvaluationResponse response = evaluator.evaluate(request);
		Assertions.assertTrue(response.isPass());
		Assertions.assertEquals(1.0, response.getScore(), 0.001);
	}
	
	@Test
	public void testBadSimilarity() {
		SimilarityEvaluator evaluator = new SimilarityEvaluator(embeddingModel, 0.95);
		EvaluationRequest request = new EvaluationRequest("the cat sat on the mat", "llms are strange");
		EvaluationResponse response = evaluator.evaluate(request);
		Assertions.assertFalse(response.isPass());
		Assertions.assertTrue(response.getScore() < 0.9);
	}
	
	@Test
	public void testReasonableSimilarity() {
		String actual = "The French capital city is Paris";
		String expected = "The capital of France is Paris";
		String badResponse = "The French capital city is Munich";
		SimilarityEvaluator evaluator = new SimilarityEvaluator(embeddingModel);
		EvaluationRequest request = new EvaluationRequest(expected, actual);
		EvaluationResponse response = evaluator.evaluate(request);
		float goodSimilarity = response.getScore();
		request = new EvaluationRequest(actual, badResponse);
		response = evaluator.evaluate(request);
		float badSimilarity = response.getScore();
		Assertions.assertTrue(goodSimilarity > badSimilarity);
	}

}
