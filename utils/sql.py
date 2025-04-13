SQL = """
WITH node_info_with_quality AS (
    SELECT ni.cache组, ni.节点名, ni.质量等级
    FROM node_info ni
    JOIN quality_level_info qli
      ON ni.省份 = qli.资源省份
    WHERE ni.节点名 IN (SELECT DISTINCT 节点名
                         FROM cache_group_info
                         WHERE cache组 IN (SELECT DISTINCT cache组
                                            FROM coverage_cache_group_info
                                            WHERE 运营商 = :operator
                                              AND 覆盖名 = :coverage_name
                                              AND IP类型 = :ip_type))
      AND qli.运营商 = :operator
      AND qli.用户省份 = :user_province
      AND ni.质量等级 <= :quality_level
)
SELECT cg.cache组, cg.节点名
FROM coverage_cache_group_info cgi
JOIN cache_group_info cg
  ON cgi.cache组 = cg.cache组
WHERE cgi.运营商 = :operator
  AND cgi.覆盖名 = :coverage_name
  AND cgi.IP类型 = :ip_type
  AND cg.节点名 IN (SELECT DISTINCT 过滤后的节点名 FROM node_info_with_quality);
"""
