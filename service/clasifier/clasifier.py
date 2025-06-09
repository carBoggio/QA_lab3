import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import logging

logger = logging.getLogger(__name__)

class KNNFaceClassifier:
    """
    Clasificador KNN para reconocimiento facial, basado en embeddings.
    """
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',  # Usar pesos basados en distancia para mejor precisión
            metric='cosine',    # Mantener métrica coseno para embeddings faciales
            n_jobs=-1          # Usar todos los cores disponibles
        )
        self.is_trained = False

    def train_from_dict(self, embeddings_dict):
        """
        Entrena el modelo KNN a partir de un diccionario de embeddings.

        Args:
            embeddings_dict (dict): 
                {
                    "persona1": [[emb1], [emb2]],
                    "persona2": [[emb3], ...],
                    ...
                }
        Returns:
            bool: True si el modelo fue entrenado correctamente
        """
        embeddings = []
        labels = []

        for label, emb_list in embeddings_dict.items():
            for emb in emb_list:
                embeddings.append(np.array(emb))
                labels.append(str(label))

        if len(embeddings) < 2:
            logger.warning("Se necesitan al menos 2 embeddings para entrenar")
            return False

        X = np.array(embeddings)
        y = np.array(labels)

        # Ajustar n_neighbors si es necesario
        if self.n_neighbors > len(X):
            logger.warning(f"Ajustando n_neighbors de {self.n_neighbors} a {len(X)}")
            self.model.n_neighbors = len(X)

        self.model.fit(X, y)
        self.is_trained = True

        logger.info(f"Modelo KNN entrenado con {len(X)} embeddings y {len(set(y))} personas")
        return True

    def predict(self, embedding):
        """
        Predice el ID de persona para un embedding usando predicción probabilística.

        Args:
            embedding (np.ndarray): Embedding facial de 512 dimensiones.

        Returns:
            str | None: ID predicho o None si no se supera el umbral de confianza.
        """
        if not self.is_trained:
            logger.warning("Modelo no entrenado")
            return None

        X = embedding.reshape(1, -1)
        
        # Obtener probabilidades de predicción
        proba = self.model.predict_proba(X)[0]
        max_proba = np.max(proba)
        
        # Obtener distancias a los vecinos más cercanos
        distances, neighbors = self.model.kneighbors(X)
        mean_distance = np.mean(distances[0])
        
        # Usar tanto probabilidad como distancia para la confianza
        confidence = max_proba * (1 - mean_distance)
        print(f"Confidence: {confidence:.4f}, Max Probability: {max_proba:.4f}, Mean Distance: {mean_distance:.4f}")
        if confidence > 0.6:
            return self.model.predict(X)[0]
        else:
            return None

# Singleton global
_knn_classifier_instance = None


def get_knn_classifier():
    global _knn_classifier_instance
    if _knn_classifier_instance is None:
        _knn_classifier_instance = KNNFaceClassifier()
    return _knn_classifier_instance
