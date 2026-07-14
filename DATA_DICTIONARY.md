# Data Dictionary

| Source | Clean name | Use |
|---|---|---|
| PatientId | patient_id | Excluded identifier |
| AppointmentID | appointment_id | Excluded identifier |
| Gender | gender | Categorical |
| ScheduledDay | scheduled_datetime | Calendar features |
| AppointmentDay | appointment_date | Calendar features and split |
| Age | age | Numeric and age group |
| Neighbourhood | neighbourhood | Categorical; governance review |
| Scholarship | scholarship | Binary |
| Hipertension | hypertension | Binary |
| Diabetes | diabetes | Binary |
| Alcoholism | alcoholism | Binary |
| Handcap | handicap | Recorded count |
| SMS_received | sms_received | Binary, observational |
| No-show | no_show | Target: Yes = 1 |
