from typing import List, Dict, Optional, Tuple
from django.shortcuts import get_object_or_404
from identidad.models import Student
from academico.models import Session
from ..pipeline_manager import get_pipeline
from identidad.serializers import process_profile_image_url
import logging

logger = logging.getLogger(__name__)


class RecognitionService:
    """
    Service class that handles facial recognition logic.
    Follows SOLID principles by separating recognition concerns from the API endpoint.
    """

    def __init__(self):
        self.pipeline = get_pipeline()
        self.similarity_threshold = 0.6
        self.possible_present_threshold = 0.4

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

        # Get enrolled students with face embeddings
        enrolled_students_data = self._get_enrolled_students_with_embeddings(session)

        if not enrolled_students_data:
            raise ValueError(
                "No hay estudiantes con embeddings faciales registrados en esta sesión"
            )

        # Process the image to get face embeddings
        face_results = self.pipeline.process_all_faces(image_file)

        # Match faces with enrolled students
        recognition_results = self._match_faces_with_students(
            face_results, enrolled_students_data, request
        )

        # Build response
        result = {
            "session_info": {
                "id": str(session.id),
                "course_name": session.course.name,
                "session_date": session.session_date.strftime("%d/%m/%Y"),
                "session_time": f"{session.start_time.strftime('%H:%M')} - {session.end_time.strftime('%H:%M')}",
            },
            "total_faces_detected": len(face_results),
            "total_enrolled_students": len(enrolled_students_data),
            "students": recognition_results["all_students"],
            "recognition_summary": {
                "recognized_count": recognition_results["recognized_count"],
                "possible_present_count": recognition_results["possible_present_count"],
                "absent_count": recognition_results["absent_count"],
                "unrecognized_faces": recognition_results["unrecognized_count"],
            },
        }

        return result

    def _get_enrolled_students_with_embeddings(
        self, session: Session
    ) -> Dict[str, Dict]:
        """
        Gets all students enrolled in the session's course that have face embeddings.

        Returns:
            Dict mapping student_id to student data including embedding
        """
        students = Student.objects.filter(
            enrollments__section=session.course
        ).prefetch_related("person_faces")

        student_data = {}

        for student in students:
            faces = student.person_faces.all()
            if faces:
                # Use the most recent embedding
                latest_face = faces.order_by("-created_at").first()
                if latest_face and latest_face.embedding is not None:
                    student_data[str(student.id)] = {
                        "student": student,
                        "embedding": latest_face.embedding,
                        "face_count": faces.count(),
                    }

        return student_data

    def _match_faces_with_students(
        self, face_results: List, enrolled_students_data: Dict, request=None
    ) -> Dict:
        """
        Matches detected faces with enrolled students.

        Args:
            face_results: List of (embedding, confidence, error) from pipeline
            enrolled_students_data: Dict of enrolled students with embeddings
            request: The HTTP request object (optional, needed for proper URL building)

        Returns:
            Dict with all students categorized by status and counts
        """
        # Track students and their best matches
        student_matches = (
            {}
        )  # student_id -> (status, confidence, face_detection_confidence)
        unrecognized_faces = 0

        for embedding, face_detection_confidence, error in face_results:
            if error:
                logger.warning(f"Error processing face: {error}")
                unrecognized_faces += 1
                continue

            if embedding is None:
                unrecognized_faces += 1
                continue

            # Find best match among enrolled students for this face
            best_student_id = None
            best_similarity = -1

            for student_id, student_data in enrolled_students_data.items():
                stored_embedding = student_data["embedding"]
                similarity = self.pipeline.recognition.compute_similarity(
                    embedding, stored_embedding
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_student_id = student_id

            # Determine status based on similarity and update student if this is a better match
            if best_student_id and best_similarity >= self.possible_present_threshold:
                # Determine status
                if best_similarity >= self.similarity_threshold:
                    status = "present"
                else:  # between 0.4 and 0.6
                    status = "possible_present"

                # Only update if this is a better match than what we already have
                if (
                    best_student_id not in student_matches
                    or student_matches[best_student_id][1] < best_similarity
                ):
                    student_matches[best_student_id] = (
                        status,
                        best_similarity,
                        face_detection_confidence,
                    )
            else:
                unrecognized_faces += 1

        # Build all students list
        all_students = []
        recognized_count = 0
        possible_present_count = 0

        # Add matched students
        for student_id, (
            status,
            confidence,
            face_detection_confidence,
        ) in student_matches.items():
            student_data = enrolled_students_data[student_id]
            student = student_data["student"]

            all_students.append(
                {
                    "id": str(student.id),
                    "name": f"{student.user.first_name} {student.user.last_name}",
                    "status": status,
                    "confidence": round(confidence, 3),
                    "face_detection_confidence": round(face_detection_confidence, 3),
                    "profile_image": process_profile_image_url(student, request),
                }
            )

            if status == "present":
                recognized_count += 1
            elif status == "possible_present":
                possible_present_count += 1

        # Add absent students (enrolled but not matched)
        for student_id, student_data in enrolled_students_data.items():
            if student_id not in student_matches:
                student = student_data["student"]
                all_students.append(
                    {
                        "id": str(student.id),
                        "name": f"{student.user.first_name} {student.user.last_name}",
                        "status": "absent",
                        "profile_image": process_profile_image_url(student, request),
                    }
                )

        absent_count = len(enrolled_students_data) - len(student_matches)

        return {
            "all_students": all_students,
            "recognized_count": recognized_count,
            "possible_present_count": possible_present_count,
            "absent_count": absent_count,
            "unrecognized_count": unrecognized_faces,
        }
