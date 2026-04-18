from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from functools import wraps

from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.tree import DecisionTreeClassifier
from werkzeug.security import check_password_hash, generate_password_hash


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


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    roll_no = db.Column(db.String(50), unique=True, nullable=False)
    branch = db.Column(db.String(50), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    attendance_records = db.relationship("Attendance", backref="student", lazy=True)
    marks = db.relationship("Marks", backref="student", lazy=True)


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
    expected_assignments = max(1, len({record.subject for record in marks})) * 2
    assignments_submitted_pct = round(
        min(100.0, (len(assignment_records) / expected_assignments) * 100), 1
    )

    subject_scores = defaultdict(list)
    for record in marks:
        if record.max_score:
            subject_scores[record.subject].append(round((record.score / record.max_score) * 100, 1))

    marks_chart = {
        "labels": list(subject_scores.keys()),
        "values": [
            round(sum(scores) / len(scores), 1) for scores in subject_scores.values()
        ],
    }

    attendance_by_month = defaultdict(lambda: {"present": 0, "total": 0})
    for record in attendance_records:
        month_label = record.date.strftime("%b %Y")
        attendance_by_month[month_label]["total"] += 1
        if record.status == "P":
            attendance_by_month[month_label]["present"] += 1

    ordered_months = sorted(
        attendance_by_month.items(),
        key=lambda item: datetime.strptime(item[0], "%b %Y"),
    )
    attendance_chart = {
        "labels": [month for month, _ in ordered_months],
        "values": [
            round((data["present"] / data["total"]) * 100, 1) if data["total"] else 0
            for _, data in ordered_months
        ],
    }

    risk_label = predict_risk(attendance_pct, avg_marks, assignments_submitted_pct)
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
        "risk_label": risk_label,
        "risk_tone": risk_tone,
    }


def predict_risk(attendance_pct: float, avg_marks: float, assignments_pct: float) -> str:
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
    prediction = model.predict([[attendance_pct, avg_marks, assignments_pct]])[0]
    return {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}[int(prediction)]


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

    db.session.commit()


@app.context_processor
def inject_globals():
    return {"current_user": current_user()}


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
        return render_template("dashboard.html", students=[], selected_student=None, metrics=None)

    requested_id = request.args.get("student_id", type=int)
    selected_student = next((student for student in students if student.id == requested_id), students[0])
    metrics = calculate_student_metrics(selected_student)
    return render_template(
        "dashboard.html",
        students=students,
        selected_student=selected_student,
        metrics=metrics,
    )


@app.route("/admin")
@login_required
@role_required("admin")
def admin_panel():
    students = Student.query.order_by(Student.name).all()
    parents = User.query.filter_by(role="parent").order_by(User.username).all()
    student_rows = []
    for student in students:
        metrics = calculate_student_metrics(student)
        student_rows.append({"student": student, "metrics": metrics})

    return render_template(
        "admin.html",
        student_rows=student_rows,
        students=students,
        parents=parents,
    )


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
    subject = request.form.get("subject", "").strip()
    status = request.form.get("status", "").strip().upper()

    if not all([student_id, date_value, subject]) or status not in {"P", "A"}:
        flash("Please provide valid attendance details.", "danger")
        return redirect(url_for("admin_panel"))

    db.session.add(
        Attendance(
            student_id=student_id,
            date=datetime.strptime(date_value, "%Y-%m-%d").date(),
            subject=subject,
            status=status,
        )
    )
    db.session.commit()
    flash("Attendance record saved.", "success")
    return redirect(url_for("admin_panel"))


@app.post("/admin/marks")
@login_required
@role_required("admin")
def add_marks():
    student_id = request.form.get("student_id", type=int)
    subject = request.form.get("subject", "").strip()
    exam_type = request.form.get("exam_type", "").strip()
    score = request.form.get("score", type=float)
    max_score = request.form.get("max_score", type=float)

    if not all([student_id, subject, exam_type]) or score is None or max_score is None or max_score <= 0:
        flash("Please provide valid marks details.", "danger")
        return redirect(url_for("admin_panel"))

    db.session.add(
        Marks(
            student_id=student_id,
            subject=subject,
            exam_type=exam_type,
            score=score,
            max_score=max_score,
        )
    )
    db.session.commit()
    flash("Marks record saved.", "success")
    return redirect(url_for("admin_panel"))


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


if __name__ == "__main__":
    app.run(debug=True)
