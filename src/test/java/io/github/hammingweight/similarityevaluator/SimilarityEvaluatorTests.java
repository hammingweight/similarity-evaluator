package io.github.hammingweight.similarityevaluator;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.evaluation.EvaluationRequest;
import org.springframework.ai.evaluation.EvaluationResponse;

import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class SimilarityEvaluatorTests {

	@Mock
	EmbeddingModel embeddingModel;

	@Test
	void cosineSimilarity() {
		// Same directions
		double cs = SimilarityEvaluator.cosineSimilarity(new float[] { 1.0f }, new float[] { 1.0f });
		Assertions.assertEquals(1.0, cs, 0.0001);
		// Opposite directions
		cs = SimilarityEvaluator.cosineSimilarity(new float[] { 1.0f }, new float[] { -2.0f });
		Assertions.assertEquals(-1.0, cs, 0.0001);
		// Orthogonal
		cs = SimilarityEvaluator.cosineSimilarity(new float[] { 1.0f, 0.0f }, new float[] { 0.0f, 1.0f });
		Assertions.assertEquals(0.0, cs, 0.0001);
		// 45 degree angle
		cs = SimilarityEvaluator.cosineSimilarity(new float[] { 1.0f, 0.0f }, new float[] { 1.0f, 1.0f });
		Assertions.assertEquals(0.707, cs, 0.001);
	}

	@Test
	void testNullEmbeddingModel() {
		Assertions.assertThrows(NullPointerException.class, () -> new SimilarityEvaluator(null));
	}

	@Test
	void testSimilarityThresholds() {
		new SimilarityEvaluator(embeddingModel);
		new SimilarityEvaluator(embeddingModel, 1.0);
		new SimilarityEvaluator(embeddingModel, -1.0);
		new SimilarityEvaluator(embeddingModel, 0.5);
		new SimilarityEvaluator(embeddingModel, 0.0);
		Assertions.assertThrows(IllegalArgumentException.class, () -> new SimilarityEvaluator(embeddingModel, 2.0));
		Assertions.assertThrows(IllegalArgumentException.class, () -> new SimilarityEvaluator(embeddingModel, -1.1));
	}

	@Test
	void testEmbeddingModelInvoked() {
		// Setup: configure mock to return embeddings for "foo" and "bar"
		when(embeddingModel.embed("foo")).thenReturn(new float[] { 1.0f, 0.0f });
		when(embeddingModel.embed("bar")).thenReturn(new float[] { 0.0f, 1.0f });

		// Create evaluator and call evaluate with expected="foo", response="bar"
		SimilarityEvaluator evaluator = new SimilarityEvaluator(embeddingModel, 0.9);
		EvaluationRequest request = new EvaluationRequest("foo", "bar");
		EvaluationResponse response = evaluator.evaluate(request);

		// Verify: embedding model was called with both texts
		verify(embeddingModel).embed("foo");
		verify(embeddingModel).embed("bar");

		// Verify: cosine similarity is 0.0 (orthogonal vectors)
		Assertions.assertEquals(0.0, response.getScore(), 0.0001);
		Assertions.assertFalse(response.isPass());
	}
}
