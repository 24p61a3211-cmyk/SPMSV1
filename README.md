# Student Performance Management System (SPMS)

## Project Overview

The Student Performance Management System (SPMS) is a web-based application designed to track and manage student academic performance, attendance, and notify parents about their children's progress. The system provides role-based access for administrators and parents to view student data, track attendance records, manage marks, and receive automated notifications about performance issues.

### Key Features

- **User Authentication**: Secure login system with role-based access control (Admin and Parent roles)
- **Student Management**: Centralized database for student information including name, roll number, branch, and year
- **Attendance Tracking**: Monitor daily attendance records for each student across different subjects
- **Marks Management**: Record and track student exam scores (mid-term, final, etc.)
- **Notifications System**: Automated alerts for low attendance and poor performance (backlog marks)
- **Admin Dashboard**: Administrative interface to manage students, subjects, and data
- **Parent Portal**: Dedicated interface for parents to view their child's performance and attendance
- **Performance Analysis**: Machine learning-based decision tree classifier for student performance prediction

### Technology Stack

- **Backend**: Flask (Python web framework)
- **Database**: SQLite with SQLAlchemy ORM
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn (Decision Tree Classification)
- **Data Import**: openpyxl (Excel file support)

---

## How to Execute the Web App

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation Steps

#### 1. Clone or Navigate to the Project

```bash
cd c:\Users\Pavan\SPMSV1
```

#### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- Flask - Web framework
- Flask-SQLAlchemy - ORM for database management
- scikit-learn - Machine learning library
- pandas - Data manipulation and analysis
- openpyxl - Excel file handling

#### 4. Run the Application

```bash
python app.py
```

#### 5. Access the Web Application

Open your web browser and navigate to:
```
http://localhost:5000
```

### Default Login Credentials

The application will create a demo database on first run with default test accounts. Please refer to the login page for demo credentials or contact your administrator.

### Project Structure

```
SPMSV1/
├── app.py              # Main Flask application and database models
├── requirements.txt    # Python package dependencies
├── instance/           # Instance folder (database files)
├── static/             # Static files
│   ├── style.css       # Application styling
│   └── js/             # JavaScript files
│       ├── admin.js    # Admin panel functionality
│       └── dashboard.js # Dashboard interactions
├── templates/          # HTML templates
│   ├── base.html       # Base template
│   ├── login.html      # Login page
│   ├── admin.html      # Admin dashboard
│   ├── dashboard.html  # Main dashboard
│   └── student.html    # Student details page
└── README.md           # This file
```

### Features Guide

#### For Administrators
- Import student data from Excel files
- Manage subjects and courses
- View overall performance metrics
- Add or update student records

#### For Parents
- View child's attendance records
- Check exam marks and performance
- Receive notifications about:
  - Low attendance (below 75%)
  - Poor marks (below 40%)
- Track performance trends

### Database

The application uses SQLite for data persistence. The database file (`spms.db`) is automatically created in the `instance/` folder on first run.

### Troubleshooting

**Issue**: Module not found errors
- **Solution**: Ensure virtual environment is activated and all dependencies are installed via `pip install -r requirements.txt`

**Issue**: Port 5000 already in use
- **Solution**: The Flask app will use the next available port or you can specify a port:
  ```bash
  python app.py --port 5001
  ```

**Issue**: Database file not found
- **Solution**: The database is created automatically on first run. Ensure the `instance/` folder has write permissions.

---

## Support

For issues, questions, or contributions, please contact the development team.

---

**Version**: 1.0  
**Last Updated**: April 2026
