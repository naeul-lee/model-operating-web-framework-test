from pydantic import BaseModel, Field

feature_names = [
    'online_daily_payment_tpv_max_90days',
       'online_daily_payment_tpv_max_30days',
       'online_daily_payment_trx_max_365days',
       'online_daily_payment_trx_mean_30days',
       'online_manwon_payment_unit_trx_365days',
       'online_daily_payment_tpv_mean_365days', 'brand_tpv_weight_365days',
       'online_daily_payment_trx_mean_365days',
       'consumption_trx_ratio_365days',
       'online_payment_method_card_trx_weight_90days',
       'online_payment_method_card_trx_weight_7days',
       'brand_trx_ratio_365days', 'brand_tpv_ratio_365days',
       'online_manwon_payment_total_tpv_30days', 'consumption_trx_365days',
       'online_payment_cycle_cycle_median_365days', 'brand_trx_weight_365days',
       'brand_tpv_180days', 'online_daily_payment_trx_max_30days',
       'consumption_age_tpv_ratio_365days', 'consumption_trx_ratio_30days',
       'consumption_trx_180days',
       'online_payment_method_card_trx_weight_30days',
       'consumption_age_trx_ratio_365days', 'consumption_trx_weight_180days',
       'consumption_trx_ratio_180days', 'online_daily_payment_trx_max_180days',
       'consumption_age_gender_trx_ratio_365days',
       'consumption_tpv_ratio_365days', 'brand_tpv_365days',
       'online_daily_payment_trx_max_7days',
       'online_daily_payment_tpv_max_365days',
       'online_daily_payment_trx_mean_180days',
       'online_payment_method_card_trx_weight_180days',
       'consumption_tpv_365days', 'consumption_age_gender_trx_ratio_180days',
       'consumption_trx_ratio_90days',
       'online_payment_cycle_cycle_mean_90days',
       'online_payment_cycle_cycle_max_365days',
       'online_daily_payment_tpv_max_180days',
       'password_cnt_password_fin_7days',
       'online_payment_cycle_cycle_mean_365days',
       'consumption_age_trx_ratio_90days', 'consumption_trx_90days',
       'online_daily_payment_trx_max_90days', 'brand_trx_30days',
       'consumption_trx_weight_30days', 'brand_trx_ratio_30days',
       'online_manwon_payment_total_tpv_max_365days', 'brand_trx_ratio_90days'
]

## ge : greater than or equal / le : less than or equal
class Features(BaseModel):
    online_daily_payment_tpv_max_90days: float = Field(..., ge=0, description="online_daily_payment_tpv_max_90days")
    online_daily_payment_tpv_max_30days: float = Field(..., ge=0, description="online_daily_payment_tpv_max_30days")
    online_daily_payment_trx_max_365days: float = Field(..., ge=0, description="online_daily_payment_trx_max_365days")
    online_daily_payment_trx_mean_30days: float = Field(..., ge=0, description="online_daily_payment_trx_mean_30days")
    online_manwon_payment_unit_trx_365days: float = Field(..., ge=0,
                                                          description="online_manwon_payment_unit_trx_365days")
    online_daily_payment_tpv_mean_365days: float = Field(..., ge=0, description="online_daily_payment_tpv_mean_365days")
    brand_tpv_weight_365days: float = Field(..., ge=0, description="brand_tpv_weight_365days")
    online_daily_payment_trx_mean_365days: float = Field(..., ge=0, description="online_daily_payment_trx_mean_365days")
    consumption_trx_ratio_365days: float = Field(..., ge=0, description="consumption_trx_ratio_365days")
    online_payment_method_card_trx_weight_90days: float = Field(..., ge=0,
                                                                description="online_payment_method_card_trx_weight_90days")
    online_payment_method_card_trx_weight_7days: float = Field(..., ge=0,
                                                               description="online_payment_method_card_trx_weight_7days")
    brand_trx_ratio_365days: float = Field(..., ge=0, description="brand_trx_ratio_365days")
    brand_tpv_ratio_365days: float = Field(..., ge=0, description="brand_tpv_ratio_365days")
    online_manwon_payment_total_tpv_30days: float = Field(..., ge=0,
                                                          description="online_manwon_payment_total_tpv_30days")
    consumption_trx_365days: float = Field(..., ge=0, description="consumption_trx_365days")
    online_payment_cycle_cycle_median_365days: float = Field(..., ge=0,
                                                             description="online_payment_cycle_cycle_median_365days")
    brand_trx_weight_365days: float = Field(..., ge=0, description="brand_trx_weight_365days")
    brand_tpv_180days: float = Field(..., ge=0, description="brand_tpv_180days")
    online_daily_payment_trx_max_30days: float = Field(..., ge=0, description="online_daily_payment_trx_max_30days")
    consumption_age_tpv_ratio_365days: float = Field(..., ge=0, description="consumption_age_tpv_ratio_365days")
    consumption_trx_ratio_30days: float = Field(..., ge=0, description="consumption_trx_ratio_30days")
    consumption_trx_180days: float = Field(..., ge=0, description="consumption_trx_180days")
    online_payment_method_card_trx_weight_30days: float = Field(..., ge=0,
                                                                description="online_payment_method_card_trx_weight_30days")
    consumption_age_trx_ratio_365days: float = Field(..., ge=0, description="consumption_age_trx_ratio_365days")
    consumption_trx_weight_180days: float = Field(..., ge=0, description="consumption_trx_weight_180days")
    consumption_trx_ratio_180days: float = Field(..., ge=0, description="consumption_trx_ratio_180days")
    online_daily_payment_trx_max_180days: float = Field(..., ge=0, description="online_daily_payment_trx_max_180days")
    consumption_age_gender_trx_ratio_365days: float = Field(..., ge=0,
                                                            description="consumption_age_gender_trx_ratio_365days")
    consumption_tpv_ratio_365days: float = Field(..., ge=0, description="consumption_tpv_ratio_365days")
    brand_tpv_365days: float = Field(..., ge=0, description="brand_tpv_365days")
    online_daily_payment_trx_max_7days: float = Field(..., ge=0, description="online_daily_payment_trx_max_7days")
    online_daily_payment_tpv_max_365days: float = Field(..., ge=0, description="online_daily_payment_tpv_max_365days")
    online_daily_payment_trx_mean_180days: float = Field(..., ge=0, description="online_daily_payment_trx_mean_180days")
    online_payment_method_card_trx_weight_180days: float = Field(..., ge=0,
                                                                 description="online_payment_method_card_trx_weight_180days")
    consumption_tpv_365days: float = Field(..., ge=0, description="consumption_tpv_365days")
    consumption_age_gender_trx_ratio_180days: float = Field(..., ge=0,
                                                            description="consumption_age_gender_trx_ratio_180days")
    consumption_trx_ratio_90days: float = Field(..., ge=0, description="consumption_trx_ratio_90days")
    online_payment_cycle_cycle_mean_90days: float = Field(..., ge=0,
                                                          description="online_payment_cycle_cycle_mean_90days")
    online_payment_cycle_cycle_max_365days: float = Field(..., ge=0,
                                                          description="online_payment_cycle_cycle_max_365days")
    online_daily_payment_tpv_max_180days: float = Field(..., ge=0, description="online_daily_payment_tpv_max_180days")
    password_cnt_password_fin_7days: float = Field(..., ge=0, description="password_cnt_password_fin_7days")
    online_payment_cycle_cycle_mean_365days: float = Field(..., ge=0,
                                                           description="online_payment_cycle_cycle_mean_365days")
    consumption_age_trx_ratio_90days: float = Field(..., ge=0, description="consumption_age_trx_ratio_90days")
    consumption_trx_90days: float = Field(..., ge=0, description="consumption_trx_90days")
    online_daily_payment_trx_max_90days: float = Field(..., ge=0, description="online_daily_payment_trx_max_90days")
    brand_trx_30days: float = Field(..., ge=0, description="brand_trx_30days")
    consumption_trx_weight_30days: float = Field(..., ge=0, description="consumption_trx_weight_30days")
    brand_trx_ratio_30days: float = Field(..., ge=0, description="brand_trx_ratio_30days")
    online_manwon_payment_total_tpv_max_365days: float = Field(..., ge=0,
                                                               description="online_manwon_payment_total_tpv_max_365days")
    brand_trx_ratio_90days: float = Field(..., ge=0, description="brand_trx_ratio_90days")


class FraudLabel(BaseModel):
    label: int = Field(
        ...,
        ge=0,
        le=1,
        description="normal tx=0 fraud = 1",
    )
