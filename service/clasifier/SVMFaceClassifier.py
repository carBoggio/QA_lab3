import numpy as np
import pickle
import os
import logging
import threading
from typing import List, Optional, Dict, Tuple
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from ..utils import load_from_database
from .Clasifier import BaseFaceClassifier

logger = logging.getLogger(__name__)


class SVMFaceClassifier(BaseFaceClassifier):
    """
    Clasificador SVM para reconocimiento facial con patrón Singleton y thread safety.
    """

    _instance = None
    _lock = threading.Lock()

    # Cargar umbrales desde environment
    THRESHOLD_PRESENTE = float(os.getenv("TRESHOLD_PRESENTE", 0.6))
    THRESHOLD_POSIBLE_PRESENTE = float(os.getenv("TRESHOLD_POSIBLE_PRESENTE", 0.3))
    logger.info(f"THRESHOLD_PRESENTE: {THRESHOLD_PRESENTE}")
    
    # Archivo de persistencia para los datos de entrenamiento
    PERSISTENCE_FILE = "svm_training_data.pkl"

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Creating new SVM Classifier instance")
                    cls._instance = super().__new__(cls)
                else:
                    logger.debug("Returning existing SVM Classifier instance")
        else:
            logger.debug("Returning existing SVM Classifier instance")
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        super().__init__()
        logger.info("Initializing SVM Classifier with RBF kernel")
        
        # Pipeline con StandardScaler y SVM
        self._model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        self._train_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._initialized = True
        logger.info("SVM Classifier initialized successfully")

    def train(self) -> bool:
        """
        Entrena el clasificador SVM cargando datos desde la base de datos y guardando en persistencia.

        Returns:
            bool: True si fue exitoso
        """
        with self._train_lock:
            try:
                logger.info("Starting SVM classifier training process...")
                # Cargar datos desde la base de datos
                embeddings_dict = load_from_database()

                if not embeddings_dict:
                    logger.error("No se pudieron cargar datos desde la base de datos")
                    raise ValueError(
                        "No se pudieron cargar datos desde la base de datos"
                    )

                logger.info(f"Loaded embeddings for {len(embeddings_dict)} students")
                for student_id, embeddings in embeddings_dict.items():
                    logger.info(f"Student {student_id}: {len(embeddings)} embeddings")

                X, y = self._prepare_training_data(embeddings_dict)

                if len(X) == 0:
                    logger.error("No hay datos de entrenamiento válidos")
                    raise ValueError("No hay datos de entrenamiento válidos")

                # Verificar que hay al menos 2 clases
                unique_classes = np.unique(y)
                if len(unique_classes) < 2:
                    logger.error("Se necesitan al menos 2 clases para entrenar el SVM")
                    raise ValueError("Se necesitan al menos 2 clases para entrenar el SVM")

                logger.info(
                    f"Training data prepared: {len(X)} samples, {len(unique_classes)} unique classes"
                )
                
                # Optimizar hiperparámetros si hay suficientes datos
                if len(X) > 20:
                    logger.info("Optimizing SVM hyperparameters...")
                    self._model = self._optimize_hyperparameters(X, y)
                else:
                    logger.info("Using default SVM parameters (insufficient data for optimization)")
                    self._model.fit(X, y)
                
                self.is_trained = True
                
                # Guardar los datos de entrenamiento en persistencia
                self._save_training_data(X, y)
                
                logger.info("SVM classifier training completed successfully")
                logger.info(f"Model info: kernel={self._model.named_steps['svm'].kernel}")
                return True

            except Exception as e:
                logger.error(f"Error durante el entrenamiento: {str(e)}")
                self.is_trained = False
                return False

    def predict(self, embedding: np.ndarray) -> Optional[Tuple[str, str]]:
        """
        Predice la persona y estado de presencia para un embedding usando datos persistentes.

        Args:
            embedding: Embedding facial

        Returns:
            Tuple[str, str]: (ID de la persona, estado_presencia) o None si no se puede predecir
        """
        with self._predict_lock:
            try:
                # Cargar datos de entrenamiento desde persistencia
                X, y = self._load_training_data()
                if X is None or y is None:
                    logger.error("No se pudieron cargar datos de entrenamiento desde persistencia")
                    return None

                if len(X) == 0:
                    logger.error("No hay datos de entrenamiento válidos en persistencia")
                    return None

                # Re-entrenar el modelo con los datos persistentes
                self._model.fit(X, y)
                self.is_trained = True

                if embedding is None or embedding.size == 0:
                    logger.warning("Embedding vacío o None recibido")
                    return None

                embedding_reshaped = embedding.reshape(1, -1)
                logger.debug(
                    f"Predicting for embedding shape: {embedding_reshaped.shape}"
                )

                # Obtener predicción y probabilidades
                prediction = self._model.predict(embedding_reshaped)
                probabilities = self._model.predict_proba(embedding_reshaped)

                # Obtener la probabilidad máxima para la predicción
                predicted_class = prediction[0]
                class_index = np.where(self._model.classes_ == predicted_class)[0][0]
                max_probability = probabilities[0][class_index]

                logger.info(
                    f"Prediction: student_id={predicted_class}, probability={max_probability:.3f}"
                )

                # Determinar estado de presencia basado en umbrales
                estado_presencia = self._determine_presence_status(max_probability)
                logger.info(f"Presence status determined: {estado_presencia}")

                return predicted_class, estado_presencia

            except Exception as e:
                logger.error(f"Error durante la predicción: {str(e)}")
                return None

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Pipeline:
        """
        Optimiza hiperparámetros usando GridSearch.
        
        Args:
            X: Datos de entrenamiento
            y: Etiquetas de entrenamiento
            
        Returns:
            Pipeline: Modelo optimizado
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Parámetros a optimizar
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
        # GridSearch con validación cruzada
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=min(3, len(np.unique(y))),  # Máximo 3-fold CV o número de clases
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        logger.info(f"Best SVM parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def _save_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Guarda los datos de entrenamiento en un archivo de persistencia.

        Args:
            X: Array de embeddings
            y: Array de labels
        """
        try:
            training_data = {
                'X': X,
                'y': y,
                'n_samples': len(X),
                'n_classes': len(np.unique(y))
            }
            
            with open(self.PERSISTENCE_FILE, 'wb') as f:
                pickle.dump(training_data, f)
            
            logger.info(f"Training data saved to {self.PERSISTENCE_FILE}: {len(X)} samples, {len(np.unique(y))} classes")
            
        except Exception as e:
            logger.error(f"Error saving training data to persistence: {str(e)}")
            raise

    def _load_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Carga los datos de entrenamiento desde el archivo de persistencia.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (X, y) o (None, None) si hay error
        """
        try:
            if not os.path.exists(self.PERSISTENCE_FILE):
                logger.error(f"Persistence file {self.PERSISTENCE_FILE} not found")
                return None, None
            
            with open(self.PERSISTENCE_FILE, 'rb') as f:
                training_data = pickle.load(f)
            
            X = training_data['X']
            y = training_data['y']
            
            logger.info(f"Training data loaded from {self.PERSISTENCE_FILE}: {len(X)} samples, {len(np.unique(y))} classes")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading training data from persistence: {str(e)}")
            return None, None

    def _determine_presence_status(self, probability: float) -> str:
        """
        Determina el estado de presencia basado en la probabilidad.

        Args:
            probability: Probabilidad de la predicción

        Returns:
            str: Estado de presencia ("presente", "posible_presente", "ausente")
        """
        if probability >= self.THRESHOLD_PRESENTE:
            return "presente"
        elif probability >= self.THRESHOLD_POSIBLE_PRESENTE:
            return "posible_presente"
        else:
            return "ausente"

    def _prepare_training_data(
        self, embeddings_dict: Dict[str, List[np.ndarray]]
    ) -> tuple:
        """
        Prepara los datos de entrenamiento para el modelo SVM.

        Args:
            embeddings_dict: Diccionario con embeddings por persona

        Returns:
            tuple: (X, y) donde X son los embeddings y y las etiquetas
        """
        X = []
        y = []

        for person_id, embeddings_list in embeddings_dict.items():
            if not embeddings_list:
                continue

            for embedding in embeddings_list:
                if embedding is not None and embedding.size > 0:
                    X.append(embedding)
                    y.append(person_id)

        if not X:
            return np.array([]), np.array([])

        return np.array(X), np.array(y)

    def get_model_info(self) -> Dict:
        """
        Obtiene información del modelo entrenado.

        Returns:
            dict: Información del modelo
        """
        if not self.is_trained:
            return {"trained": False}

        svm_model = self._model.named_steps['svm']
        return {
            "trained": True,
            "kernel": svm_model.kernel,
            "C": svm_model.C,
            "gamma": svm_model.gamma,
            "n_support_vectors": getattr(svm_model, "n_support_", []).tolist() if hasattr(svm_model, "n_support_") else [],
            "classes": getattr(svm_model, "classes_", []).tolist(),
            "n_classes": len(getattr(svm_model, "classes_", [])),
        }


def get_svm_classifier() -> SVMFaceClassifier:
    """
    Obtiene la instancia singleton del clasificador SVM.
    Siempre retorna la misma instancia debido al patrón Singleton implementado en SVMFaceClassifier.

    Returns:
        SVMFaceClassifier: Instancia singleton del clasificador
    """
    instance = SVMFaceClassifier()
    logger.debug(f"SVM Classifier instance ID: {id(instance)}")
    return instance