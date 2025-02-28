-- 实验组/对照组用户统计与续费率分析
WITH user_groups AS (
    SELECT 
        u.user_id,
        t.group AS teacher_group
    FROM users u
    LEFT JOIN teachers t ON u.experience_teacher_id = t.teacher_id
),

call_stats AS (
    SELECT
        ug.teacher_group,
        COUNT(DISTINCT cl.user_id) AS called_users,
        AVG(cl.call_duration) AS avg_duration
    FROM user_groups ug
    LEFT JOIN call_logs cl ON ug.user_id = cl.user_id
    GROUP BY ug.teacher_group
),

renewal_stats AS (
    SELECT
        ug.teacher_group,
        COUNT(DISTINCT CASE WHEN r.renewal_status = 1 THEN ug.user_id END) AS renewed_users
    FROM user_groups ug
    LEFT JOIN renewals r ON ug.user_id = r.user_id
    GROUP BY ug.teacher_group
)

SELECT
    ug.teacher_group,
    COUNT(DISTINCT ug.user_id) AS total_users,
    ROUND(COALESCE(cs.called_users * 1.0 / COUNT(DISTINCT ug.user_id), 4) AS call_coverage_rate,
    ROUND(COALESCE(cs.avg_duration, 0), 1) AS avg_call_duration,
    ROUND(COALESCE(rs.renewed_users * 1.0 / COUNT(DISTINCT ug.user_id), 0), 4) AS renewal_rate
FROM user_groups ug
LEFT JOIN call_stats cs ON ug.teacher_group = cs.teacher_group
LEFT JOIN renewal_stats rs ON ug.teacher_group = rs.teacher_group
GROUP BY ug.teacher_group;





