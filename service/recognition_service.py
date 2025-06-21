from typing import List, Dict, Optional, Tuple
from django.shortcuts import get_object_or_404
from identidad.models import Student
from academico.models import Session
from ..pipeline_manager import get_pipeline
from identidad.serializers import process_profile_image_url
from .BaseFaceRecognitionPipeline import BaseFaceRecognitionPipeline
import logging

logger = logging.getLogger(__name__)


class RecognitionService:
    def __init__(self):
        self.pipeline: BaseFaceRecognitionPipeline = get_pipeline()

    def recognize_students_in_session(
        self, image_file, session_id: str, request=None
    ) -> Dict:
        """
        Recognizes students in an image for a specific session.

        Args:
            image_file: The uploaded image file
            session_id: The session ID to get enrolled students from
            request: The HTTP request object (optional, needed for proper URL building)

        Returns:
            Dict with recognition results including recognized students and metrics
        """
        # Verify session exists
        session = get_object_or_404(Session, id=session_id)

        # Get enrolled students
        enrolled_students = self._get_enrolled_students(session)

        if not enrolled_students:
            raise ValueError("No hay estudiantes inscritos en esta sesión")

        # Use the new pipeline method to predict all people identities
        recognition_results = self.pipeline.predict_people_identity_from_picture(image_file)
        logger.info(f"Pipeline predicted {len(recognition_results)} people in image")

        # Match predictions with enrolled students
        matched_results = self._match_predictions_with_students(
            recognition_results, enrolled_students, request
        )
        logger.info(f"Matched results: {matched_results}")

        # Build response
        result = {
            "session_info": {
                "id": str(session.id),
                "course_name": session.course.name,
                "session_date": session.session_date.strftime("%d/%m/%Y"),
                "session_time": f"{session.start_time.strftime('%H:%M')} - {session.end_time.strftime('%H:%M')}",
            },
            "total_faces_detected": len(recognition_results),
            "total_enrolled_students": len(enrolled_students),
            "students": matched_results["all_students"],
            "recognition_summary": {
                "recognized_count": matched_results["recognized_count"],
                "possible_present_count": matched_results["possible_present_count"],
                "absent_count": matched_results["absent_count"],
                "unrecognized_faces": matched_results["unrecognized_count"],
            },
        }

        return result

    def _get_enrolled_students(self, session: Session) -> Dict[str, Student]:
        """
        Gets all students enrolled in the session's course.
        """
        students = Student.objects.filter(
            enrollments__section=session.course
        ).distinct()  # Evita duplicados
        return {str(student.id): student for student in students}

    def _match_predictions_with_students(
        self, recognition_results: List[Tuple[str, str]], enrolled_students: Dict[str, Student], request=None
    ) -> Dict:
        """
        Matches pipeline predictions with enrolled students.
        """
        identified_students = {}  # Dict[student_id, presence_status]
        unrecognized_faces = 0

        for person_name, status in recognition_results:
            if person_name == "no_identificado" or person_name == "presente":
                # Person detected but not identified or not enrolled
                unrecognized_faces += 1
                logger.info(f"❓ Face detected but person not identified: {person_name}")
                continue

            # Check if the predicted person is enrolled in this session
            if person_name in enrolled_students:
                # Student was identified and is enrolled in this session
                if person_name in identified_students:
                    # Si el estudiante ya fue identificado, actualizamos su estado solo si es más favorable
                    current_status = identified_students[person_name]
                    # Definimos el orden de prioridad: presente > posible_presente > ausente
                    status_priority = {
                        "presente": 2,
                        "posible_presente": 1,
                        "ausente": 0,
                    }
                    if (
                        status_priority[status]
                        > status_priority[current_status]
                    ):
                        identified_students[person_name] = status
                else:
                    identified_students[person_name] = status
                logger.info(
                    f"✅ Student identified: {person_name} with status {status}"
                )
            else:
                # Person identified but not enrolled in this session
                unrecognized_faces += 1
                logger.info(
                    f"⚠️ Person {person_name} not enrolled in this session"
                )

        # Build all students list
        all_students = []
        recognized_count = len(
            [s for s in identified_students.values() if s == "presente"]
        )
        possible_present_count = len(
            [s for s in identified_students.values() if s == "posible_presente"]
        )

        # Add identified students with their presence status
        for student_id, presence_status in identified_students.items():
            student = enrolled_students[student_id]
            status = (
                "present"
                if presence_status == "presente"
                else "possible_present"
                if presence_status == "posible_presente"
                else "absent"
            )
            all_students.append(
                {
                    "id": str(student.id),
                    "name": f"{student.user.first_name} {student.user.last_name}",
                    "status": status,
                    "method": "facial_recognition",
                    "profile_image": process_profile_image_url(student, request),
                }
            )

        # Add absent students (enrolled but not identified)
        for student_id, student in enrolled_students.items():
            if student_id not in identified_students:
                all_students.append(
                    {
                        "id": str(student.id),
                        "name": f"{student.user.first_name} {student.user.last_name}",
                        "status": "absent",
                        "profile_image": process_profile_image_url(student, request),
                    }
                )

        absent_count = (
            len(enrolled_students) - recognized_count - possible_present_count
        )

        return {
            "all_students": all_students,
            "recognized_count": recognized_count,
            "possible_present_count": possible_present_count,
            "absent_count": absent_count,
            "unrecognized_count": unrecognized_faces,
        }

    def get_detailed_session_analysis(self, image_file, session_id: str) -> Dict:
        """
        Método adicional que aprovecha las nuevas capacidades del pipeline.
        Proporciona análisis más detallado usando los nuevos métodos.
        """
        session = get_object_or_404(Session, id=session_id)
        enrolled_students = self._get_enrolled_students(session)

        # Usar el método de predicción del nuevo pipeline
        recognition_results = self.pipeline.predict_people_identity_from_picture(image_file)

        # Usar el método de dibujo de caras
        annotated_image = self.pipeline.draw_faces_in_picture(image_file)

        return {
            "session_id": str(session.id),
            "total_enrolled": len(enrolled_students),
            "face_analysis": {
                "faces_detected": len(recognition_results),
                "recognition_results": recognition_results,
            },
            "face_count_verification": len(recognition_results),
            "pipeline_status": {
                "annotated_image_available": annotated_image is not None,
            },
        }

    def verify_single_student_attendance(
        self, image_file, student_id: str, session_id: str
    ) -> Dict:
        """
        Verifica si un estudiante específico está presente en la imagen.
        Usa el nuevo método de extracción de embedding del pipeline.
        """
        session = get_object_or_404(Session, id=session_id)
        student = get_object_or_404(Student, id=student_id)

        # Verificar si el estudiante está inscrito en la sesión
        is_enrolled = Student.objects.filter(
            id=student_id, enrollments__section=session.course
        ).exists()

        if not is_enrolled:
            return {
                "student_id": student_id,
                "is_present": False,
                "reason": "student_not_enrolled",
                "confidence": "sin_datos",
            }

        # Extraer embedding de la cara más grande en la imagen
        embedding = self.pipeline.extract_embedding_from_single_largest_face_image(image_file)
        
        if embedding is None:
            return {
                "student_id": student_id,
                "is_present": False,
                "reason": "no_face_detected",
                "confidence": "sin_datos",
            }

        # Usar el clasificador para predecir la identidad
        prediction_result = self.pipeline.classifier.predict(embedding)
        
        if prediction_result is None:
            return {
                "student_id": student_id,
                "is_present": False,
                "reason": "not_identified",
                "confidence": "sin_datos",
            }

        predicted_student_id, presence_status = prediction_result
        
        is_present = (predicted_student_id == student_id and presence_status == "presente")
        confidence = presence_status if predicted_student_id == student_id else "wrong_person"

        return {
            "student_id": student_id,
            "student_name": f"{student.user.first_name} {student.user.last_name}",
            "is_present": is_present,
            "confidence": confidence,
            "session_id": str(session.id),
        }

    def retrain_pipeline_classifier(self) -> Dict:
        """
        Fuerza el re-entrenamiento del clasificador del pipeline.
        Útil cuando se agregan nuevos estudiantes.
        """
        try:
            success = self.pipeline.classifier.train()
            return {
                "retrain_successful": success,
                "message": (
                    "Clasificador re-entrenado exitosamente"
                    if success
                    else "Error en re-entrenamiento"
                ),
            }
        except Exception as e:
            logger.error(f"Error retraining classifier: {str(e)}")
            return {"retrain_successful": False, "message": f"Error: {str(e)}"}

    def get_annotated_image(self, image_file) -> bytes:
        """
        Obtiene la imagen con las caras dibujadas usando el nuevo pipeline.
        """
        try:
            annotated_image = self.pipeline.draw_faces_in_picture(image_file)
            return annotated_image
        except Exception as e:
            logger.error(f"Error getting annotated image: {str(e)}")
            return None
