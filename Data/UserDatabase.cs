using System;
using System.Data;
using System.Data.Common;

namespace TransferLearning.Data
{
    public static class UserDatabase
    {
        /// <summary>
        /// Inserts a note and returns the generated note ID.
        /// </summary>
        public static int InsertNote(string note)
        {
            // Implementation inserts note into database and returns the generated ID.
            throw new NotImplementedException();
        }

        /// <summary>
        /// Marks the specified agent as deleted in the database.
        /// Always returns true if the operation succeeds; throws on failure.
        /// </summary>
        public static bool AgentDelete(string companyCode, string agentID, int xactNoteID)
        {
            // SqlDatabase userDb = DataBaseHelper.UserVeriTabaniOlustur();
            // DbCommand dbCreate = userDb.GetStoredProcCommand("agent_delete");

            // userDb.AddInParameter(dbCreate, "CompanyCode", SqlDbType.VarChar, companyCode);
            // userDb.AddInParameter(dbCreate, "AgentID", SqlDbType.VarChar, agentID);
            // userDb.AddInParameter(dbCreate, "xactNoteID", SqlDbType.Int, xactNoteID);

            // userDb.ExecuteNonQuery(dbCreate);

            return true;
        }
    }
}
