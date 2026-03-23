using System;
using System.Web.Mvc;
using TransferLearning.Models;
using TransferLearning.Services;

namespace TransferLearning.Controllers
{
    public class AgentController : Controller
    {
        /// <summary>
        /// Deletes an agent identified by the encrypted <paramref name="agentIdEnc"/> within
        /// the company identified by the encrypted <paramref name="companyCodeEnc"/>.
        /// Returns a JSON result indicating success or failure.
        /// </summary>
        public JsonResult AgentDelete(string companyCodeEnc, string agentIdEnc, string note)
        {
            var sonuc = new IslemSonucu();

            string companyCode = GenelHelper.DencryptString(companyCodeEnc);
            string agentID = GenelHelper.DencryptString(agentIdEnc);

            if (string.IsNullOrEmpty(companyCode) || string.IsNullOrEmpty(agentID))
                return Json(
                    new { IsSuccess = sonuc.IslemBasarili, MyException = "Parametre hatası." },
                    JsonRequestBehavior.AllowGet);

            var userDataService = new UserDataService(companyCode);

            try
            {
                var oldAgent = userDataService.GetAgentByIDFromDb(agentID);

                sonuc.IslemBasarili = userDataService.AgentDelete(agentID, note);

                ChangeLog.WriteChangeLog(
                    companyCode,
                    companyCode == agentID ? ObjectTypes.COMPANY : ObjectTypes.AGENT,
                    agentID,
                    TransactionTypes.DELETE,
                    null,
                    null,
                    this.BOMKullanici.Email);

                if (sonuc.IslemBasarili)
                {
                    var paynetVService = new PaynetVService();
                    var syncResult = paynetVService.SyncMerchant(companyCode, agentID);

                    if (!syncResult)
                        ErrorLog.Yaz(
                            Modul.PAYNETBOM,
                            new Exception(
                                $"Müşteri PaynetV'te güncellenemedi. CompanyCode: {companyCode} AgentId: {agentID}"),
                            logLevel: LogSeverityLevel.Warning);

                    SyncAgentInReportDb(agentID);

                    if (oldAgent != null && !oldAgent.IsAgentDeleted)
                    {
                        var customerApi = new CustomerApiService();
                        customerApi.SendMerchantAlarm(new SendMerchantAlarmParameters
                        {
                            agent_id = agentID,
                            date = DateTime.Now,
                            modifyUser = this.BOMKullanici.Email,
                            type = SendMerchantAlarmType.Delete.GetEnumDescription(),
                            username = "-",
                            note = note
                        });
                    }
                }
            }
            catch (AgentAlreadyDeletedException ex)
            {
                // Agent is already in deleted status — surface the reason to the caller
                // without logging a full error since this is an expected business-rule violation.
                sonuc.IslemBasarili = false;
                sonuc.Mesaj = ex.Message;
            }
            catch (Exception ex)
            {
                ErrorLog.Yaz(Modul.PAYNETBOM, ex, "BOM", BOMKullanici.Email, null);
                sonuc.IslemBasarili = false;
                sonuc.Mesaj = "Bir hata oluştu.";
            }

            return Json(
                new { IsSuccess = sonuc.IslemBasarili, MyException = sonuc.Mesaj },
                JsonRequestBehavior.AllowGet);
        }

        private void SyncAgentInReportDb(string agentID)
        {
            throw new NotImplementedException();
        }
    }
}
