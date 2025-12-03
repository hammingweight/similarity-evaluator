package io.github.hammingweight.similarityevaluator;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.embedding.EmbeddingModel;

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
}
