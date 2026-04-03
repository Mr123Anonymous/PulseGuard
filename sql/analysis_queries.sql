-- 1) Cohort size and positive readmission rate
SELECT
    COUNT(*) AS total_rows,
    AVG(CASE WHEN target_readmit_30d = 1 THEN 1.0 ELSE 0.0 END) AS readmit_30d_rate
FROM admissions;

-- 2) Readmission rate by age band
SELECT
    age,
    COUNT(*) AS n,
    AVG(CASE WHEN target_readmit_30d = 1 THEN 1.0 ELSE 0.0 END) AS readmit_30d_rate
FROM admissions
GROUP BY age
ORDER BY age;

-- 3) Readmission by admission type
SELECT
    admission_type_id,
    COUNT(*) AS n,
    AVG(CASE WHEN target_readmit_30d = 1 THEN 1.0 ELSE 0.0 END) AS readmit_30d_rate
FROM admissions
GROUP BY admission_type_id
ORDER BY n DESC;

-- 4) Length-of-stay bucket impact
SELECT
    CASE
        WHEN time_in_hospital <= 3 THEN '0-3 days'
        WHEN time_in_hospital <= 7 THEN '4-7 days'
        ELSE '8+ days'
    END AS los_bucket,
    COUNT(*) AS n,
    AVG(CASE WHEN target_readmit_30d = 1 THEN 1.0 ELSE 0.0 END) AS readmit_30d_rate
FROM admissions
GROUP BY
    CASE
        WHEN time_in_hospital <= 3 THEN '0-3 days'
        WHEN time_in_hospital <= 7 THEN '4-7 days'
        ELSE '8+ days'
    END
ORDER BY n DESC;

-- 5) Top diagnosis groups by readmission burden
SELECT
    diag_1,
    COUNT(*) AS n,
    SUM(CASE WHEN target_readmit_30d = 1 THEN 1 ELSE 0 END) AS readmit_count
FROM admissions
GROUP BY diag_1
HAVING COUNT(*) >= 100
ORDER BY readmit_count DESC
LIMIT 20;

-- 6) Medication count and readmission trend
SELECT
    num_medications,
    COUNT(*) AS n,
    AVG(CASE WHEN target_readmit_30d = 1 THEN 1.0 ELSE 0.0 END) AS readmit_30d_rate
FROM admissions
GROUP BY num_medications
HAVING COUNT(*) >= 50
ORDER BY num_medications;
