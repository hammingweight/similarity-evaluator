package io.github.hammingweight.similarityevaluator;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.evaluation.EvaluationRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.test.context.SpringBootTest;

import io.github.hammingweight.similarityevaluator.SimilarityEvaluator.SimilarityEvaluationResponse;

@SpringBootTest
@SpringBootApplication
public class SimilarityEvaluatorIntegrationTests {
	
	@Autowired
	public EmbeddingModel embeddingModel;

	
	@Test
	public void testGoodSimilarity() {
		SimilarityEvaluator evaluator = new SimilarityEvaluator(embeddingModel);
		EvaluationRequest request = new EvaluationRequest("the cat sat on the mat", "the cat sat on the mat");
		SimilarityEvaluationResponse response = evaluator.evaluate(request);
		Assertions.assertTrue(response.isPass());
		Assertions.assertEquals(1.0, response.getCosineSimilarity(), 0.001);
	}
	
	@Test
	public void testBadSimilarity() {
		SimilarityEvaluator evaluator = new SimilarityEvaluator(embeddingModel, 0.95);
		EvaluationRequest request = new EvaluationRequest("the cat sat on the mat", "llms are strange");
		SimilarityEvaluationResponse response = evaluator.evaluate(request);
		Assertions.assertFalse(response.isPass());
		Assertions.assertTrue(response.getCosineSimilarity() < 0.9);
	}
	
	@Test
	public void testReasonableSimilarity() {
		String actual = "The French capital city is Paris";
		String expected = "The capital of France is Paris";
		String badResponse = "The French capital city is Munich";
		SimilarityEvaluator evaluator = new SimilarityEvaluator(embeddingModel);
		EvaluationRequest request = new EvaluationRequest(expected, actual);
		SimilarityEvaluationResponse response = evaluator.evaluate(request);
		double goodSimilarity = response.getCosineSimilarity();
		request = new EvaluationRequest(actual, badResponse);
		response = evaluator.evaluate(request);
		double badSimilarity = response.getCosineSimilarity();
		Assertions.assertTrue(goodSimilarity > badSimilarity);
	}

}
