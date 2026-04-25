from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from functools import lru_cache, wraps
from io import BytesIO
import json
from pathlib import Path
import secrets
import string

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import UniqueConstraint, func
from werkzeug.security import check_password_hash, generate_password_hash


LOW_ATTENDANCE_THRESHOLD = 75.0
BACKLOG_MARKS_THRESHOLD = 40.0

IMPORT_REQUIRED_SHEETS = {
    "Students": {"roll_no", "name", "branch", "year", "parent_username", "subjects"},
    "Attendance": {"roll_no", "date", "subject", "status"},
    "Marks": {"roll_no", "subject", "exam_type", "score", "max_score"},
}
IMPORT_OPTIONAL_STUDENT_COLUMNS = {"parent_email"}


app = Flask(__name__)
app.config["SECRET_KEY"] = "spms-demo-secret-key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///spms.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    students = db.relationship("Student", backref="parent", lazy=True)
    notifications = db.relationship("Notification", backref="user", lazy=True)


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    roll_no = db.Column(db.String(50), unique=True, nullable=False)
    branch = db.Column(db.String(50), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    attendance_records = db.relationship("Attendance", backref="student", lazy=True)
    marks = db.relationship("Marks", backref="student", lazy=True)
    subject_links = db.relationship("StudentSubject", backref="student", lazy=True)
    notifications = db.relationship("Notification", backref="student", lazy=True)


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("student.id"), nullable=False)
    date = db.Column(db.Date, nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(1), nullable=False)


class Marks(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("student.id"), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    exam_type = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Float, nullable=False)
    max_score = db.Column(db.Float, nullable=False)


class Subject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    student_links = db.relationship("StudentSubject", backref="subject", lazy=True)


class StudentSubject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("student.id"), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey("subject.id"), nullable=False)

    __table_args__ = (UniqueConstraint("student_id", "subject_id", name="uq_student_subject"),)


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey("student.id"), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(160), nullable=False)
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), nullable=False, default="info")
    is_read = db.Column(db.Boolean, nullable=False, default=False)
    is_resolved = db.Column(db.Boolean, nullable=False, default=False)
    event_key = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)

    return wrapped_view


def role_required(role):
    def decorator(view):
        @wraps(view)
        def wrapped_view(*args, **kwargs):
            user = current_user()
            if not user or user.role != role:
                flash("You do not have permission to access that page.", "warning")
                return redirect(url_for("login_page"))
            return view(*args, **kwargs)

        return wrapped_view

    return decorator


def current_user() -> User | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.session.get(User, user_id)


def normalize_subject_name(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def parse_import_date(value: str) -> datetime.date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def generate_temp_password(length: int = 10) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@lru_cache(maxsize=1)
def risk_model() -> DecisionTreeClassifier:
    training_rows = []
    training_labels = []
    for attendance in range(50, 101, 5):
        for marks in range(30, 101, 5):
            assignment = min(100, max(40, int((attendance + marks) / 2)))
            if attendance < 75 or marks < 40:
                label = 2
            elif attendance < 85 or marks < 60:
                label = 1
            else:
                label = 0
            training_rows.append([attendance, marks, assignment])
            training_labels.append(label)

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(training_rows, training_labels)
    return model


def predict_risk(attendance_pct: float, avg_marks: float, assignments_pct: float) -> str:
    prediction = risk_model().predict([[attendance_pct, avg_marks, assignments_pct]])[0]
    return {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}[int(prediction)]


def calculate_risk_score(attendance_pct: float, avg_marks: float, assignments_pct: float) -> float:
    weighted_risk = (100 - attendance_pct) * 0.4 + (100 - avg_marks) * 0.4 + (100 - assignments_pct) * 0.2
    return round(max(0.0, min(100.0, weighted_risk)), 1)


def get_or_create_subject(subject_name: str) -> Subject | None:
    cleaned_name = normalize_subject_name(subject_name)
    if not cleaned_name:
        return None

    subject = Subject.query.filter(func.lower(Subject.name) == cleaned_name.lower()).first()
    if subject:
        return subject

    subject = Subject(name=cleaned_name)
    db.session.add(subject)
    db.session.flush()
    return subject


def ensure_student_subject(student: Student, subject_name: str) -> None:
    subject = get_or_create_subject(subject_name)
    if not subject:
        return
    existing = StudentSubject.query.filter_by(student_id=student.id, subject_id=subject.id).first()
    if not existing:
        db.session.add(StudentSubject(student_id=student.id, subject_id=subject.id))


def sync_subjects_for_student(student: Student, subject_names: list[str] | set[str]) -> None:
    for subject_name in subject_names:
        ensure_student_subject(student, subject_name)


def calculate_student_metrics(student: Student) -> dict:
    attendance_records = student.attendance_records
    marks = student.marks

    total_classes = len(attendance_records)
    present_classes = sum(1 for record in attendance_records if record.status == "P")
    attendance_pct = round((present_classes / total_classes) * 100, 1) if total_classes else 0.0

    mark_percentages = [
        (record.score / record.max_score) * 100 for record in marks if record.max_score
    ]
    avg_marks = round(sum(mark_percentages) / len(mark_percentages), 1) if mark_percentages else 0.0
    gpa = round((avg_marks / 10), 2) if avg_marks else 0.0

    assignment_records = [record for record in marks if record.exam_type.lower() == "assignment"]
    expected_assignments = max(1, len({normalize_subject_name(record.subject) for record in marks})) * 2
    assignments_submitted_pct = round(
        min(100.0, (len(assignment_records) / expected_assignments) * 100), 1
    )

    subject_scores: defaultdict[str, list[float]] = defaultdict(list)
    for record in marks:
        if record.max_score:
            subject_name = normalize_subject_name(record.subject)
            if subject_name:
                subject_scores[subject_name].append(round((record.score / record.max_score) * 100, 1))

    subject_averages = [
        {"subject": subject, "avg": round(sum(scores) / len(scores), 1)}
        for subject, scores in subject_scores.items()
    ]
    subject_averages.sort(key=lambda item: item["subject"])
    backlog_subjects = [item for item in subject_averages if item["avg"] < BACKLOG_MARKS_THRESHOLD]

    marks_chart = {
        "labels": [item["subject"] for item in subject_averages],
        "values": [item["avg"] for item in subject_averages],
    }

    attendance_by_month: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"present": 0, "absent": 0, "total": 0}
    )
    for record in attendance_records:
        month_label = record.date.strftime("%b %Y")
        attendance_by_month[month_label]["total"] += 1
        if record.status == "P":
            attendance_by_month[month_label]["present"] += 1
        else:
            attendance_by_month[month_label]["absent"] += 1

    ordered_months = sorted(
        attendance_by_month.items(),
        key=lambda item: datetime.strptime(item[0], "%b %Y"),
    )
    attendance_chart = {
        "labels": [month for month, _ in ordered_months],
        "present": [data["present"] for _, data in ordered_months],
        "absent": [data["absent"] for _, data in ordered_months],
    }
    attendance_table = [
        {
            "month": month,
            "present": data["present"],
            "absent": data["absent"],
            "total": data["total"],
            "pct": round((data["present"] / data["total"]) * 100, 1) if data["total"] else 0.0,
        }
        for month, data in ordered_months
    ]

    risk_label = predict_risk(attendance_pct, avg_marks, assignments_submitted_pct)
    risk_score = calculate_risk_score(attendance_pct, avg_marks, assignments_submitted_pct)
    risk_tone = {
        "Low Risk": "success",
        "Medium Risk": "warning",
        "High Risk": "danger",
    }[risk_label]

    return {
        "attendance_pct": attendance_pct,
        "avg_marks": avg_marks,
        "gpa": gpa,
        "assignments_submitted_pct": assignments_submitted_pct,
        "marks_chart": marks_chart,
        "attendance_chart": attendance_chart,
        "attendance_table": attendance_table,
        "risk_label": risk_label,
        "risk_score": risk_score,
        "risk_tone": risk_tone,
        "backlog_subjects": backlog_subjects,
        "backlog_count": len(backlog_subjects),
    }


def get_notification_recipients(student: Student) -> list[User]:
    recipients: dict[int, User] = {}
    if student.parent:
        recipients[student.parent.id] = student.parent

    admins = User.query.filter_by(role="admin").all()
    for admin in admins:
        recipients[admin.id] = admin
    return list(recipients.values())


def create_notification(
    *,
    user_id: int,
    student_id: int,
    notification_type: str,
    title: str,
    message: str,
    severity: str,
    event_key: str | None = None,
) -> Notification:
    notification = Notification(
        user_id=user_id,
        student_id=student_id,
        type=notification_type,
        title=title,
        message=message,
        severity=severity,
        event_key=event_key,
    )
    db.session.add(notification)
    return notification


def upsert_condition_notification(
    *,
    user_id: int,
    student_id: int,
    notification_type: str,
    title: str,
    message: str,
    severity: str,
    event_key: str,
) -> None:
    existing = (
        Notification.query.filter_by(user_id=user_id, event_key=event_key, is_resolved=False)
        .order_by(Notification.created_at.desc())
        .first()
    )
    if existing:
        content_changed = (
            existing.title != title
            or existing.message != message
            or existing.severity != severity
        )
        existing.title = title
        existing.message = message
        existing.severity = severity
        existing.type = notification_type
        existing.updated_at = datetime.utcnow()
        if content_changed:
            existing.is_read = False
        return

    create_notification(
        user_id=user_id,
        student_id=student_id,
        notification_type=notification_type,
        title=title,
        message=message,
        severity=severity,
        event_key=event_key,
    )


def resolve_condition_notification(user_id: int, event_key: str) -> None:
    unresolved = Notification.query.filter_by(
        user_id=user_id,
        event_key=event_key,
        is_resolved=False,
    ).all()
    for notification in unresolved:
        notification.is_resolved = True
        notification.updated_at = datetime.utcnow()


def unread_notifications_count(user_id: int) -> int:
    return Notification.query.filter_by(
        user_id=user_id,
        is_read=False,
        is_resolved=False,
    ).count()


def get_user_notifications(user_id: int, limit: int = 50) -> list[Notification]:
    return (
        Notification.query.filter_by(user_id=user_id, is_resolved=False)
        .order_by(Notification.is_read.asc(), Notification.created_at.desc())
        .limit(limit)
        .all()
    )


def evaluate_condition_notifications(student: Student, metrics: dict | None = None) -> None:
    if metrics is None:
        metrics = calculate_student_metrics(student)

    recipients = get_notification_recipients(student)
    low_attendance_key = f"student:{student.id}:low_attendance"
    backlog_key = f"student:{student.id}:backlog"
    risk_key = f"student:{student.id}:risk"

    for recipient in recipients:
        if metrics["attendance_pct"] < LOW_ATTENDANCE_THRESHOLD:
            upsert_condition_notification(
                user_id=recipient.id,
                student_id=student.id,
                notification_type="low_attendance",
                title=f"Low attendance: {student.name}",
                message=(
                    f"{student.name} is at {metrics['attendance_pct']}% attendance. "
                    f"Required minimum is {LOW_ATTENDANCE_THRESHOLD}%."
                ),
                severity="warning",
                event_key=low_attendance_key,
            )
        else:
            resolve_condition_notification(recipient.id, low_attendance_key)

        backlog_subjects = metrics["backlog_subjects"]
        if backlog_subjects:
            subjects_text = ", ".join(item["subject"] for item in backlog_subjects[:4])
            if len(backlog_subjects) > 4:
                subjects_text += ", ..."

            upsert_condition_notification(
                user_id=recipient.id,
                student_id=student.id,
                notification_type="backlog",
                title=f"Backlog risk: {student.name}",
                message=(
                    f"{student.name} has {len(backlog_subjects)} subject(s) below "
                    f"{BACKLOG_MARKS_THRESHOLD}% average: {subjects_text}."
                ),
                severity="danger",
                event_key=backlog_key,
            )
        else:
            resolve_condition_notification(recipient.id, backlog_key)

        if metrics["risk_label"] in {"Medium Risk", "High Risk"}:
            severity = "danger" if metrics["risk_label"] == "High Risk" else "warning"
            upsert_condition_notification(
                user_id=recipient.id,
                student_id=student.id,
                notification_type="risk",
                title=f"{metrics['risk_label']}: {student.name}",
                message=(
                    f"Risk score is {metrics['risk_score']} / 100 "
                    f"({metrics['risk_label']}). Attendance: {metrics['attendance_pct']}%, "
                    f"average marks: {metrics['avg_marks']}%."
                ),
                severity=severity,
                event_key=risk_key,
            )
        else:
            resolve_condition_notification(recipient.id, risk_key)


def create_new_score_notifications(mark_record: Marks) -> None:
    recipients = get_notification_recipients(mark_record.student)
    percentage = (
        round((mark_record.score / mark_record.max_score) * 100, 1)
        if mark_record.max_score
        else 0.0
    )
    for recipient in recipients:
        create_notification(
            user_id=recipient.id,
            student_id=mark_record.student_id,
            notification_type="new_score",
            title=f"New score: {mark_record.student.name}",
            message=(
                f"{mark_record.subject} ({mark_record.exam_type}) score is "
                f"{mark_record.score}/{mark_record.max_score} ({percentage}%)."
            ),
            severity="info",
            event_key=None,
        )


def refresh_student_alerts(student: Student) -> None:
    metrics = calculate_student_metrics(student)
    evaluate_condition_notifications(student, metrics)


def normalize_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip().lower() for column in cleaned.columns]
    return cleaned


def parse_students_sheet(frame: pd.DataFrame) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []

    for idx, raw_row in frame.iterrows():
        row_number = idx + 2
        roll_no = str(raw_row.get("roll_no", "")).strip()
        name = str(raw_row.get("name", "")).strip()
        branch = str(raw_row.get("branch", "")).strip()
        parent_username = str(raw_row.get("parent_username", "")).strip()
        parent_email = str(raw_row.get("parent_email", "")).strip()

        year_raw = raw_row.get("year")
        year_numeric = pd.to_numeric(year_raw, errors="coerce")
        year = int(year_numeric) if pd.notna(year_numeric) else None

        subjects_raw = str(raw_row.get("subjects", "")).strip()
        subjects = [
            normalize_subject_name(item)
            for item in subjects_raw.split(",")
            if normalize_subject_name(item)
        ]

        if not roll_no:
            errors.append(f"Students row {row_number}: roll_no is required.")
        if not name:
            errors.append(f"Students row {row_number}: name is required.")
        if not branch:
            errors.append(f"Students row {row_number}: branch is required.")
        if year is None:
            errors.append(f"Students row {row_number}: year must be a number.")
        elif year < 1 or year > 6:
            errors.append(f"Students row {row_number}: year must be between 1 and 6.")
        if not parent_username:
            errors.append(f"Students row {row_number}: parent_username is required.")

        if roll_no and name and branch and year is not None and parent_username:
            rows.append(
                {
                    "roll_no": roll_no,
                    "name": name,
                    "branch": branch,
                    "year": year,
                    "parent_username": parent_username,
                    "parent_email": parent_email,
                    "subjects": subjects,
                }
            )

    seen_roll_nos: set[str] = set()
    for row in rows:
        if row["roll_no"] in seen_roll_nos:
            errors.append(f"Students sheet: duplicate roll_no '{row['roll_no']}'.")
        seen_roll_nos.add(row["roll_no"])

    return rows, errors


def parse_attendance_sheet(frame: pd.DataFrame) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []

    for idx, raw_row in frame.iterrows():
        row_number = idx + 2
        roll_no = str(raw_row.get("roll_no", "")).strip()
        subject = normalize_subject_name(raw_row.get("subject", ""))
        status = str(raw_row.get("status", "")).strip().upper()
        date_raw = raw_row.get("date")
        parsed_date = pd.to_datetime(date_raw, errors="coerce")

        if not roll_no:
            errors.append(f"Attendance row {row_number}: roll_no is required.")
        if pd.isna(parsed_date):
            errors.append(f"Attendance row {row_number}: date must be YYYY-MM-DD.")
        if not subject:
            errors.append(f"Attendance row {row_number}: subject is required.")
        if status not in {"P", "A"}:
            errors.append(f"Attendance row {row_number}: status must be P or A.")

        if roll_no and pd.notna(parsed_date) and subject and status in {"P", "A"}:
            rows.append(
                {
                    "roll_no": roll_no,
                    "date": parsed_date.strftime("%Y-%m-%d"),
                    "subject": subject,
                    "status": status,
                }
            )

    return rows, errors


def parse_marks_sheet(frame: pd.DataFrame) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []

    for idx, raw_row in frame.iterrows():
        row_number = idx + 2
        roll_no = str(raw_row.get("roll_no", "")).strip()
        subject = normalize_subject_name(raw_row.get("subject", ""))
        exam_type = str(raw_row.get("exam_type", "")).strip()
        score_numeric = pd.to_numeric(raw_row.get("score"), errors="coerce")
        max_score_numeric = pd.to_numeric(raw_row.get("max_score"), errors="coerce")

        score = float(score_numeric) if pd.notna(score_numeric) else None
        max_score = float(max_score_numeric) if pd.notna(max_score_numeric) else None

        if not roll_no:
            errors.append(f"Marks row {row_number}: roll_no is required.")
        if not subject:
            errors.append(f"Marks row {row_number}: subject is required.")
        if not exam_type:
            errors.append(f"Marks row {row_number}: exam_type is required.")
        if score is None:
            errors.append(f"Marks row {row_number}: score must be numeric.")
        elif score < 0:
            errors.append(f"Marks row {row_number}: score cannot be negative.")
        if max_score is None or max_score <= 0:
            errors.append(f"Marks row {row_number}: max_score must be greater than 0.")
        if score is not None and max_score is not None and score > max_score:
            errors.append(f"Marks row {row_number}: score cannot exceed max_score.")

        if (
            roll_no
            and subject
            and exam_type
            and score is not None
            and score >= 0
            and max_score
            and score <= max_score
        ):
            rows.append(
                {
                    "roll_no": roll_no,
                    "subject": subject,
                    "exam_type": exam_type,
                    "score": score,
                    "max_score": max_score,
                }
            )

    return rows, errors


def parse_import_workbook(file_storage) -> tuple[dict, list[str]]:
    payload = {"students": [], "attendance": [], "marks": []}
    errors: list[str] = []

    file_bytes = file_storage.read()
    if not file_bytes:
        return payload, ["Uploaded file is empty."]

    try:
        workbook = pd.read_excel(BytesIO(file_bytes), sheet_name=None, engine="openpyxl")
    except Exception as exc:
        return payload, [f"Could not parse workbook: {exc}"]

    for sheet_name, required_columns in IMPORT_REQUIRED_SHEETS.items():
        if sheet_name not in workbook:
            errors.append(f"Missing required sheet: '{sheet_name}'.")
            continue

        frame = normalize_frame_columns(workbook[sheet_name])
        available_columns = set(frame.columns)
        optional_columns = IMPORT_OPTIONAL_STUDENT_COLUMNS if sheet_name == "Students" else set()
        missing_columns = required_columns - (available_columns | optional_columns)
        if missing_columns:
            missing_columns_text = ", ".join(sorted(missing_columns))
            errors.append(
                f"Sheet '{sheet_name}' is missing required column(s): {missing_columns_text}."
            )
            continue

        if sheet_name == "Students":
            rows, row_errors = parse_students_sheet(frame)
            payload["students"] = rows
        elif sheet_name == "Attendance":
            rows, row_errors = parse_attendance_sheet(frame)
            payload["attendance"] = rows
        else:
            rows, row_errors = parse_marks_sheet(frame)
            payload["marks"] = rows
        errors.extend(row_errors)

    known_roll_nos = {row["roll_no"] for row in payload["students"]}
    known_roll_nos.update(roll_no for (roll_no,) in db.session.query(Student.roll_no).all())

    for row in payload["attendance"]:
        if row["roll_no"] not in known_roll_nos:
            errors.append(
                f"Attendance references unknown roll_no '{row['roll_no']}'. "
                "Add this student in Students sheet or existing database."
            )
    for row in payload["marks"]:
        if row["roll_no"] not in known_roll_nos:
            errors.append(
                f"Marks references unknown roll_no '{row['roll_no']}'. "
                "Add this student in Students sheet or existing database."
            )

    return payload, errors


def get_import_preview_dir() -> Path:
    directory = Path(app.instance_path) / "import_previews"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def preview_file_path(token: str) -> Path:
    return get_import_preview_dir() / f"{token}.json"


def save_import_preview(payload: dict) -> str:
    token = secrets.token_urlsafe(18)
    with preview_file_path(token).open("w", encoding="utf-8") as preview_file:
        json.dump(payload, preview_file)
    return token


def load_import_preview(token: str) -> dict | None:
    if not token:
        return None
    path = preview_file_path(token)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as preview_file:
        return json.load(preview_file)


def delete_import_preview(token: str) -> None:
    if not token:
        return
    path = preview_file_path(token)
    if path.exists():
        path.unlink()


def get_or_create_parent_for_import(parent_username: str) -> tuple[User | None, str | None, str | None]:
    existing_user = User.query.filter_by(username=parent_username).first()
    if existing_user:
        if existing_user.role != "parent":
            return None, None, f"Username '{parent_username}' exists but is not a parent account."
        return existing_user, None, None

    temporary_password = generate_temp_password()
    parent = User(
        username=parent_username,
        password=generate_password_hash(temporary_password),
        role="parent",
    )
    db.session.add(parent)
    db.session.flush()
    return parent, temporary_password, None


def apply_import_payload(payload: dict) -> dict:
    result = {
        "students_inserted": 0,
        "students_updated": 0,
        "attendance_inserted": 0,
        "attendance_updated": 0,
        "marks_inserted": 0,
        "marks_updated": 0,
        "parents_created": [],
        "errors": [],
    }

    roll_cache: dict[str, Student] = {}
    affected_student_ids: set[int] = set()
    new_marks_records: list[Marks] = []

    for row in payload.get("students", []):
        parent, temp_password, parent_error = get_or_create_parent_for_import(
            row["parent_username"]
        )
        if parent_error:
            result["errors"].append(parent_error)
            continue

        student = Student.query.filter_by(roll_no=row["roll_no"]).first()
        is_new_student = student is None
        if is_new_student:
            student = Student(
                name=row["name"],
                roll_no=row["roll_no"],
                branch=row["branch"],
                year=row["year"],
                parent_id=parent.id,
            )
            db.session.add(student)
            db.session.flush()
            result["students_inserted"] += 1
        else:
            student.name = row["name"]
            student.branch = row["branch"]
            student.year = row["year"]
            student.parent_id = parent.id
            result["students_updated"] += 1

        roll_cache[row["roll_no"]] = student
        affected_student_ids.add(student.id)
        sync_subjects_for_student(student, row.get("subjects", []))

        if temp_password:
            result["parents_created"].append(
                {"username": parent.username, "temporary_password": temp_password}
            )

    db.session.flush()

    def resolve_student_by_roll(roll_no: str) -> Student | None:
        if roll_no in roll_cache:
            return roll_cache[roll_no]
        student_lookup = Student.query.filter_by(roll_no=roll_no).first()
        if student_lookup:
            roll_cache[roll_no] = student_lookup
        return student_lookup

    for row in payload.get("attendance", []):
        student = resolve_student_by_roll(row["roll_no"])
        if not student:
            result["errors"].append(
                f"Attendance skipped: roll_no '{row['roll_no']}' was not found."
            )
            continue

        date_value = parse_import_date(row["date"])
        subject_name = normalize_subject_name(row["subject"])
        existing = (
            Attendance.query.filter_by(
                student_id=student.id,
                date=date_value,
                subject=subject_name,
            )
            .order_by(Attendance.id.asc())
            .first()
        )

        if existing:
            existing.status = row["status"]
            result["attendance_updated"] += 1
        else:
            db.session.add(
                Attendance(
                    student_id=student.id,
                    date=date_value,
                    subject=subject_name,
                    status=row["status"],
                )
            )
            result["attendance_inserted"] += 1

        sync_subjects_for_student(student, [subject_name])
        affected_student_ids.add(student.id)

    for row in payload.get("marks", []):
        student = resolve_student_by_roll(row["roll_no"])
        if not student:
            result["errors"].append(f"Marks skipped: roll_no '{row['roll_no']}' was not found.")
            continue

        subject_name = normalize_subject_name(row["subject"])
        existing = (
            Marks.query.filter_by(
                student_id=student.id,
                subject=subject_name,
                exam_type=row["exam_type"],
            )
            .order_by(Marks.id.asc())
            .first()
        )
        if existing:
            existing.score = row["score"]
            existing.max_score = row["max_score"]
            result["marks_updated"] += 1
        else:
            mark_row = Marks(
                student_id=student.id,
                subject=subject_name,
                exam_type=row["exam_type"],
                score=row["score"],
                max_score=row["max_score"],
            )
            db.session.add(mark_row)
            new_marks_records.append(mark_row)
            result["marks_inserted"] += 1

        sync_subjects_for_student(student, [subject_name])
        affected_student_ids.add(student.id)

    db.session.flush()

    for mark_row in new_marks_records:
        create_new_score_notifications(mark_row)

    for student_id in affected_student_ids:
        student = db.session.get(Student, student_id)
        if student:
            refresh_student_alerts(student)

    db.session.commit()
    return result


def build_import_preview(filename: str, payload: dict, errors: list[str], token: str | None = None) -> dict:
    return {
        "filename": filename,
        "token": token,
        "errors": errors,
        "summary": {
            "students": len(payload.get("students", [])),
            "attendance": len(payload.get("attendance", [])),
            "marks": len(payload.get("marks", [])),
        },
        "samples": {
            "Students": payload.get("students", [])[:5],
            "Attendance": payload.get("attendance", [])[:5],
            "Marks": payload.get("marks", [])[:5],
        },
    }


def build_admin_context(
    import_preview: dict | None = None,
    import_result: dict | None = None,
) -> dict:
    students = Student.query.order_by(Student.name).all()
    parents = User.query.filter_by(role="parent").order_by(User.username).all()
    user = current_user()

    risk_distribution = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
    low_attendance_students = []
    backlog_students = []
    student_rows = []

    for student in students:
        metrics = calculate_student_metrics(student)
        student_rows.append({"student": student, "metrics": metrics})
        risk_distribution[metrics["risk_label"]] += 1

        if metrics["attendance_pct"] < LOW_ATTENDANCE_THRESHOLD:
            low_attendance_students.append(
                {
                    "name": student.name,
                    "roll_no": student.roll_no,
                    "attendance_pct": metrics["attendance_pct"],
                }
            )

        if metrics["backlog_count"]:
            backlog_students.append(
                {
                    "name": student.name,
                    "roll_no": student.roll_no,
                    "subjects": [item["subject"] for item in metrics["backlog_subjects"]],
                    "count": metrics["backlog_count"],
                }
            )

    low_attendance_students.sort(key=lambda row: row["attendance_pct"])
    backlog_students.sort(key=lambda row: row["count"], reverse=True)

    notifications = get_user_notifications(user.id, limit=80) if user else []
    admin_payload = {
        "riskDistribution": {
            "labels": ["Low Risk", "Medium Risk", "High Risk"],
            "values": [
                risk_distribution["Low Risk"],
                risk_distribution["Medium Risk"],
                risk_distribution["High Risk"],
            ],
        }
    }

    return {
        "student_rows": student_rows,
        "students": students,
        "parents": parents,
        "notifications": notifications,
        "low_attendance_students": low_attendance_students,
        "backlog_students": backlog_students,
        "admin_payload": admin_payload,
        "import_preview": import_preview,
        "import_result": import_result,
    }


def seed_demo_data() -> None:
    if User.query.first():
        return

    admin = User(
        username="admin",
        password=generate_password_hash("admin123"),
        role="admin",
    )
    parent_one = User(
        username="parent_ravi",
        password=generate_password_hash("parent123"),
        role="parent",
    )
    parent_two = User(
        username="parent_sneha",
        password=generate_password_hash("parent123"),
        role="parent",
    )
    parent_three = User(
        username="parent_arun",
        password=generate_password_hash("parent123"),
        role="parent",
    )
    db.session.add_all([admin, parent_one, parent_two, parent_three])
    db.session.flush()

    students = [
        Student(name="Rahul Verma", roll_no="CSE101", branch="CSE", year=2, parent_id=parent_one.id),
        Student(name="Diya Iyer", roll_no="ECE205", branch="ECE", year=3, parent_id=parent_two.id),
        Student(name="Kiran Patel", roll_no="ME310", branch="Mechanical", year=4, parent_id=parent_three.id),
        Student(name="Aarav Nair", roll_no="CSE119", branch="CSE", year=1, parent_id=parent_one.id),
    ]
    db.session.add_all(students)
    db.session.flush()

    student_subject_map: defaultdict[int, set[str]] = defaultdict(set)

    attendance_rows = [
        (students[0], [("2026-01-10", "Mathematics", "P"), ("2026-01-12", "Physics", "P"), ("2026-02-02", "Programming", "P"), ("2026-02-18", "Data Structures", "P"), ("2026-03-05", "Mathematics", "A"), ("2026-03-12", "Programming", "P"), ("2026-04-04", "Physics", "P"), ("2026-04-11", "Data Structures", "P")]),
        (students[1], [("2026-01-10", "Signals", "P"), ("2026-01-12", "Networks", "P"), ("2026-02-02", "Embedded Systems", "A"), ("2026-02-18", "Digital Logic", "P"), ("2026-03-05", "Signals", "P"), ("2026-03-12", "Networks", "A"), ("2026-04-04", "Embedded Systems", "P"), ("2026-04-11", "Digital Logic", "P")]),
        (students[2], [("2026-01-10", "Thermodynamics", "A"), ("2026-01-12", "Machine Design", "A"), ("2026-02-02", "Manufacturing", "P"), ("2026-02-18", "CAD", "A"), ("2026-03-05", "Thermodynamics", "P"), ("2026-03-12", "Machine Design", "A"), ("2026-04-04", "Manufacturing", "P"), ("2026-04-11", "CAD", "A")]),
        (students[3], [("2026-01-10", "Maths I", "P"), ("2026-01-12", "English", "P"), ("2026-02-02", "Python", "P"), ("2026-02-18", "Electronics", "P"), ("2026-03-05", "Maths I", "P"), ("2026-03-12", "Python", "P"), ("2026-04-04", "English", "P"), ("2026-04-11", "Electronics", "P")]),
    ]
    for student, records in attendance_rows:
        for date_value, subject, status in records:
            db.session.add(
                Attendance(
                    student_id=student.id,
                    date=datetime.strptime(date_value, "%Y-%m-%d").date(),
                    subject=subject,
                    status=status,
                )
            )
            student_subject_map[student.id].add(subject)

    marks_rows = [
        (students[0], [("Mathematics", "Mid-term", 78, 100), ("Physics", "Mid-term", 74, 100), ("Programming", "Assignment", 18, 20), ("Data Structures", "Final", 82, 100), ("Programming", "Final", 85, 100), ("Physics", "Assignment", 17, 20)]),
        (students[1], [("Signals", "Mid-term", 61, 100), ("Networks", "Mid-term", 58, 100), ("Embedded Systems", "Assignment", 16, 20), ("Digital Logic", "Final", 64, 100), ("Signals", "Final", 55, 100), ("Networks", "Assignment", 14, 20)]),
        (students[2], [("Thermodynamics", "Mid-term", 34, 100), ("Machine Design", "Mid-term", 39, 100), ("Manufacturing", "Assignment", 9, 20), ("CAD", "Final", 42, 100), ("Thermodynamics", "Final", 38, 100), ("Machine Design", "Assignment", 8, 20)]),
        (students[3], [("Maths I", "Mid-term", 88, 100), ("English", "Mid-term", 79, 100), ("Python", "Assignment", 19, 20), ("Electronics", "Final", 83, 100), ("Python", "Final", 91, 100), ("English", "Assignment", 18, 20)]),
    ]
    for student, records in marks_rows:
        for subject, exam_type, score, max_score in records:
            db.session.add(
                Marks(
                    student_id=student.id,
                    subject=subject,
                    exam_type=exam_type,
                    score=score,
                    max_score=max_score,
                )
            )
            student_subject_map[student.id].add(subject)

    db.session.flush()
    for student in students:
        sync_subjects_for_student(student, student_subject_map[student.id])

    db.session.commit()


def bootstrap_existing_subjects_and_notifications() -> None:
    students = Student.query.all()
    for student in students:
        subject_names = {
            normalize_subject_name(record.subject)
            for record in student.attendance_records + student.marks
            if normalize_subject_name(record.subject)
        }
        sync_subjects_for_student(student, subject_names)
        refresh_student_alerts(student)
    db.session.commit()


@app.context_processor
def inject_globals():
    user = current_user()
    unread_count = unread_notifications_count(user.id) if user else 0
    return {
        "current_user": user,
        "unread_notifications_count": unread_count,
    }


@app.route("/")
def login_page():
    user = current_user()
    if user:
        if user.role == "admin":
            return redirect(url_for("admin_panel"))
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.post("/login")
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    user = User.query.filter_by(username=username).first()

    if not user or not check_password_hash(user.password, password):
        flash("Invalid username or password.", "danger")
        return redirect(url_for("login_page"))

    session["user_id"] = user.id
    flash(f"Welcome back, {user.username}.", "success")
    if user.role == "admin":
        return redirect(url_for("admin_panel"))
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login_page"))


@app.route("/dashboard")
@login_required
@role_required("parent")
def dashboard():
    user = current_user()
    students = Student.query.filter_by(parent_id=user.id).order_by(Student.name).all()
    if not students:
        return render_template(
            "dashboard.html",
            students=[],
            selected_student=None,
            metrics=None,
            notifications=get_user_notifications(user.id),
            dashboard_payload={},
        )

    requested_id = request.args.get("student_id", type=int)
    selected_student = next((student for student in students if student.id == requested_id), students[0])
    metrics = calculate_student_metrics(selected_student)
    dashboard_payload = {
        "marks": metrics["marks_chart"],
        "attendance": metrics["attendance_chart"],
        "risk": {
            "label": metrics["risk_label"],
            "score": metrics["risk_score"],
            "tone": metrics["risk_tone"],
        },
    }

    return render_template(
        "dashboard.html",
        students=students,
        selected_student=selected_student,
        metrics=metrics,
        notifications=get_user_notifications(user.id),
        dashboard_payload=dashboard_payload,
    )


@app.route("/admin")
@login_required
@role_required("admin")
def admin_panel():
    return render_template("admin.html", **build_admin_context())


@app.post("/admin/student")
@login_required
@role_required("admin")
def add_student():
    name = request.form.get("name", "").strip()
    roll_no = request.form.get("roll_no", "").strip()
    branch = request.form.get("branch", "").strip()
    year = request.form.get("year", type=int)
    parent_id = request.form.get("parent_id", type=int)

    if not all([name, roll_no, branch, year, parent_id]):
        flash("Please fill all student fields.", "danger")
        return redirect(url_for("admin_panel"))

    existing = Student.query.filter_by(roll_no=roll_no).first()
    if existing:
        flash("Roll number already exists.", "warning")
        return redirect(url_for("admin_panel"))

    db.session.add(
        Student(name=name, roll_no=roll_no, branch=branch, year=year, parent_id=parent_id)
    )
    db.session.commit()
    flash("Student added successfully.", "success")
    return redirect(url_for("admin_panel"))


@app.post("/admin/attendance")
@login_required
@role_required("admin")
def add_attendance():
    student_id = request.form.get("student_id", type=int)
    date_value = request.form.get("date", "").strip()
    subject = normalize_subject_name(request.form.get("subject", ""))
    status = request.form.get("status", "").strip().upper()

    if not all([student_id, date_value, subject]) or status not in {"P", "A"}:
        flash("Please provide valid attendance details.", "danger")
        return redirect(url_for("admin_panel"))

    try:
        parsed_date = datetime.strptime(date_value, "%Y-%m-%d").date()
    except ValueError:
        flash("Date must be in YYYY-MM-DD format.", "danger")
        return redirect(url_for("admin_panel"))

    attendance_row = Attendance(
        student_id=student_id,
        date=parsed_date,
        subject=subject,
        status=status,
    )
    db.session.add(attendance_row)
    student = db.session.get(Student, student_id)
    if student:
        sync_subjects_for_student(student, [subject])
        db.session.flush()
        refresh_student_alerts(student)

    db.session.commit()
    flash("Attendance record saved.", "success")
    return redirect(url_for("admin_panel"))


@app.post("/admin/marks")
@login_required
@role_required("admin")
def add_marks():
    student_id = request.form.get("student_id", type=int)
    subject = normalize_subject_name(request.form.get("subject", ""))
    exam_type = request.form.get("exam_type", "").strip()
    score = request.form.get("score", type=float)
    max_score = request.form.get("max_score", type=float)

    if (
        not all([student_id, subject, exam_type])
        or score is None
        or score < 0
        or max_score is None
        or max_score <= 0
        or score > max_score
    ):
        flash("Please provide valid marks details.", "danger")
        return redirect(url_for("admin_panel"))

    marks_row = Marks(
        student_id=student_id,
        subject=subject,
        exam_type=exam_type,
        score=score,
        max_score=max_score,
    )
    db.session.add(marks_row)
    student = db.session.get(Student, student_id)
    if student:
        sync_subjects_for_student(student, [subject])
    db.session.flush()
    create_new_score_notifications(marks_row)
    if student:
        refresh_student_alerts(student)

    db.session.commit()
    flash("Marks record saved and notifications updated.", "success")
    return redirect(url_for("admin_panel"))


@app.post("/admin/import/preview")
@login_required
@role_required("admin")
def admin_import_preview():
    workbook = request.files.get("workbook")
    if not workbook or not workbook.filename:
        flash("Please upload an Excel workbook (.xlsx).", "danger")
        return redirect(url_for("admin_panel"))

    payload, errors = parse_import_workbook(workbook)
    preview = build_import_preview(workbook.filename, payload, errors)

    if errors:
        flash("Import preview has validation errors. Fix them and upload again.", "warning")
        return render_template("admin.html", **build_admin_context(import_preview=preview))

    token = save_import_preview(payload)
    preview["token"] = token
    flash("Import preview generated. Review and confirm to apply changes.", "success")
    return render_template("admin.html", **build_admin_context(import_preview=preview))


@app.post("/admin/import/confirm")
@login_required
@role_required("admin")
def admin_import_confirm():
    token = request.form.get("preview_token", "").strip()
    payload = load_import_preview(token)
    if not payload:
        flash("Import preview expired or not found. Upload file again.", "warning")
        return redirect(url_for("admin_panel"))

    try:
        import_result = apply_import_payload(payload)
    except Exception as exc:
        db.session.rollback()
        delete_import_preview(token)
        flash(f"Import failed: {exc}", "danger")
        return redirect(url_for("admin_panel"))

    delete_import_preview(token)

    if import_result["errors"]:
        flash(
            "Import completed with warnings. Review skipped rows in the report below.",
            "warning",
        )
    else:
        flash("Import completed successfully.", "success")

    return render_template("admin.html", **build_admin_context(import_result=import_result))


@app.post("/notifications/<int:notification_id>/read")
@login_required
def mark_notification_read(notification_id: int):
    notification = db.session.get(Notification, notification_id)
    user = current_user()
    if not notification or notification.user_id != user.id:
        flash("Notification not found.", "warning")
        return redirect(request.referrer or url_for("login_page"))

    notification.is_read = True
    notification.updated_at = datetime.utcnow()
    db.session.commit()
    return redirect(request.referrer or url_for("login_page"))


@app.post("/notifications/read-all")
@login_required
def mark_all_notifications_read():
    user = current_user()
    unread_notifications = Notification.query.filter_by(
        user_id=user.id,
        is_read=False,
        is_resolved=False,
    ).all()
    for notification in unread_notifications:
        notification.is_read = True
        notification.updated_at = datetime.utcnow()
    db.session.commit()
    return redirect(request.referrer or url_for("login_page"))


@app.route("/student/<int:student_id>")
@login_required
def student_detail(student_id: int):
    student = db.session.get(Student, student_id)
    if not student:
        flash("Student not found.", "warning")
        return redirect(url_for("login_page"))

    user = current_user()
    if user.role == "parent" and student.parent_id != user.id:
        flash("You can only view your own child.", "warning")
        return redirect(url_for("dashboard"))

    metrics = calculate_student_metrics(student)
    attendance_records = Attendance.query.filter_by(student_id=student.id).order_by(Attendance.date.desc()).all()
    marks = Marks.query.filter_by(student_id=student.id).order_by(Marks.subject, Marks.exam_type).all()
    return render_template(
        "student.html",
        student=student,
        metrics=metrics,
        attendance_records=attendance_records,
        marks=marks,
    )


with app.app_context():
    db.create_all()
    seed_demo_data()
    bootstrap_existing_subjects_and_notifications()


if __name__ == "__main__":
    app.run(debug=True)
