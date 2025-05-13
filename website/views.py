from flask import Blueprint, render_template, jsonify
from flask_login import login_required, current_user
import json
from . import db
from .models import Report
from flask import request, jsonify
views = Blueprint('views', __name__)

@views.route('/')
@login_required
def home():
    reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.date.desc()).all()

    parsed_reports = []
    for report in reports:
        try:
            emotion_data = json.loads(report.data)
        except json.JSONDecodeError:
            emotion_data = {"Invalid": "Data"}
        parsed_reports.append({
            "id": report.id,
            "data": emotion_data,
            "date": report.date.strftime("%B %d, %Y %H:%M")
        })

    return render_template("home.html", user=current_user, reports=parsed_reports)



@views.route('/delete-report', methods=['POST'])
def delete_report():  
    report = json.loads(request.data) # this function expects a JSON from the INDEX.js file 
    reportId = report['reportId']
    report = Report.query.get(reportId)
    if report:
        if report.user_id == current_user.id:
            db.session.delete(report)
            db.session.commit()

    return jsonify({})