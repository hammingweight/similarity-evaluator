package io.github.hammingweight.similarityevaluator;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.evaluation.EvaluationRequest;
import org.springframework.ai.evaluation.EvaluationResponse;
import org.springframework.ai.evaluation.Evaluator;
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
		Evaluator evaluator = new SimilarityEvaluator(embeddingModel);

		EvaluationRequest request = new EvaluationRequest("the cat sat on the mat", "the cat sat on the mat");
		EvaluationResponse response = evaluator.evaluate(request);

		Assertions.assertTrue(response.isPass());
		Assertions.assertEquals(1.0, response.getScore(), 0.001);
	}

	@Test
	public void testBadSimilarity() {
		Evaluator evaluator = new SimilarityEvaluator(embeddingModel, 0.95);

		EvaluationRequest request = new EvaluationRequest("the cat sat on the mat", "llms are strange");
		EvaluationResponse response = evaluator.evaluate(request);

		Assertions.assertFalse(response.isPass());
		Assertions.assertTrue(response.getScore() < 0.9);
	}

	@Test
	public void testReasonableSimilarity() {
		Evaluator evaluator = new SimilarityEvaluator(embeddingModel);

		String expected = "The French capital city is Paris";
		String goodResponse = "The capital of France is Paris";
		String badResponse = "The French capital city is Munich";

		EvaluationRequest goodRequest = new EvaluationRequest(expected, goodResponse);
		EvaluationResponse good = evaluator.evaluate(goodRequest);

		EvaluationRequest badRequest = new EvaluationRequest(expected, badResponse);
		EvaluationResponse bad = evaluator.evaluate(badRequest);

		Assertions.assertTrue(good.getScore() > bad.getScore());
	}

}
