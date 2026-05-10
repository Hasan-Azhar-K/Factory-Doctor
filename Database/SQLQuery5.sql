-- ============================================================
--  setup_database.sql  --  Factory Doctor FYP
--
--  HOW TO USE:
--    1. Open SQL Server Management Studio (SSMS)
--    2. Connect to your local SQL Server
--    3. Open this file and press F5 (Execute)
--
--  Safe to run multiple times -- tables only created if they
--  do not already exist. Views are always re-created cleanly.
-- ============================================================


-- ─── Step 1: Create the database ────────────────────────────
USE master;
GO

IF NOT EXISTS (
    SELECT name FROM sys.databases WHERE name = 'FactoryDoctorDB'
)
BEGIN
    CREATE DATABASE FactoryDoctorDB;
    PRINT '>> Database FactoryDoctorDB created.';
END
ELSE
BEGIN
    PRINT '>> Database FactoryDoctorDB already exists. Skipping.';
END
GO

USE FactoryDoctorDB;
GO


-- ─── Step 2: Create MachineData table ───────────────────────
--  One row per sensor reading. Exactly matches what Python inserts.
IF NOT EXISTS (
    SELECT 1
    FROM   sys.tables
    WHERE  name      = 'MachineData'
    AND    schema_id = SCHEMA_ID('dbo')
)
BEGIN
    CREATE TABLE dbo.MachineData
    (
        ID                  INT           IDENTITY(1,1) NOT NULL,
        MachineID           VARCHAR(10)                 NOT NULL,
        AirTemperature      FLOAT                       NOT NULL,
        ProcessTemperature  FLOAT                       NOT NULL,
        RotationalSpeed     FLOAT                       NOT NULL,
        Torque              FLOAT                       NOT NULL,
        ToolWear            FLOAT                       NOT NULL,
        FailureProbability  FLOAT                       NOT NULL,
        HealthStatus        VARCHAR(10)                 NOT NULL,
        RecordedAt          DATETIME                    NOT NULL DEFAULT GETDATE(),

        CONSTRAINT PK_MachineData PRIMARY KEY (ID)
    );

    -- Speeds up per-machine time-ordered queries
    CREATE INDEX IX_MachineData_Machine_Time
        ON dbo.MachineData (MachineID, RecordedAt DESC);

    PRINT '>> Table dbo.MachineData created.';
END
ELSE
BEGIN
    PRINT '>> Table dbo.MachineData already exists. Skipping.';
END
GO


-- ─── Step 3: View -- latest reading per machine ─────────────
--  Drop before (re)creating so this is safe on every run.
IF OBJECT_ID('dbo.vw_LatestReadings', 'V') IS NOT NULL
    DROP VIEW dbo.vw_LatestReadings;
GO

CREATE VIEW dbo.vw_LatestReadings
AS
SELECT m.*
FROM   dbo.MachineData AS m
INNER JOIN
(
    SELECT   MachineID,
             MAX(RecordedAt) AS LatestTime
    FROM     dbo.MachineData
    GROUP BY MachineID
) AS latest
    ON  m.MachineID  = latest.MachineID
    AND m.RecordedAt = latest.LatestTime
GO

PRINT '>> View dbo.vw_LatestReadings created.';
GO


-- ─── Step 4: View -- 24-hour health summary per machine ─────
IF OBJECT_ID('dbo.vw_HealthSummary24h', 'V') IS NOT NULL
    DROP VIEW dbo.vw_HealthSummary24h;
GO

CREATE VIEW dbo.vw_HealthSummary24h
AS
SELECT
    MachineID,
    HealthStatus,
    COUNT(*)                AS ReadingCount,
    AVG(FailureProbability) AS AvgFailureProb,
    MAX(FailureProbability) AS MaxFailureProb,
    AVG(ToolWear)           AS AvgToolWear,
    MIN(RecordedAt)         AS PeriodStart,
    MAX(RecordedAt)         AS PeriodEnd
FROM  dbo.MachineData
WHERE RecordedAt >= DATEADD(HOUR, -24, GETDATE())
GROUP BY
    MachineID,
    HealthStatus
GO

PRINT '>> View dbo.vw_HealthSummary24h created.';
GO


-- ─── Done ────────────────────────────────────────────────────
PRINT '';
PRINT '==============================================';
PRINT '  Setup complete. Now run your Python files.  ';
PRINT '==============================================';
GO


-- ─── Verification queries (uncomment and run manually) ───────
/*

-- Check table columns match what Python inserts
SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_NAME = 'MachineData'
ORDER BY ORDINAL_POSITION;

-- Row count per machine (use after simulation starts)
SELECT   MachineID, COUNT(*) AS TotalRows
FROM     dbo.MachineData
GROUP BY MachineID
ORDER BY MachineID;

-- Latest reading per machine
SELECT * FROM dbo.vw_LatestReadings
ORDER BY MachineID;

-- 24-hour summary
SELECT * FROM dbo.vw_HealthSummary24h
ORDER BY MachineID, HealthStatus;

-- Critical events in the last hour
SELECT MachineID, FailureProbability, ToolWear, RecordedAt
FROM   dbo.MachineData
WHERE  HealthStatus = 'Critical'
AND    RecordedAt  >= DATEADD(HOUR, -1, GETDATE())
ORDER BY RecordedAt DESC;

*/
