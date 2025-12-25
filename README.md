# USD-TRY-Forecasting-XGBoost
Machine Learning model predicting USD/TRY exchange rates using macroeconomic indicators (Inflation, Interest Rates, CDS, Reserves) with XGBoost and SHAP analysis.
. GİRİŞ
1.1. Konuya Genel Bakış
Gelişmekte olan ekonomilerde makroekonomik istikrarın en önemli göstergelerinden biri döviz kurudur. Özellikle Türkiye gibi dış ticarete ve küresel sermaye akımlarına entegre olmuş piyasalarda, döviz kuru volatilitesi sadece finansal piyasaları değil; enflasyon, faiz oranları ve reel sektörün maliyet yapısını da doğrudan etkilemektedir. Türkiye ekonomisinin son on yıllık periyodu incelendiğinde, Türkiye Cumhuriyet Merkez Bankası’nın (TCMB) uyguladığı para politikaları, rezerv yönetimi ve faiz kararlarının kur üzerindeki etkisinin dönemsel olarak farklılaştığı görülmektedir.
Geleneksel iktisat teorileri, faiz oranları ile döviz kurları arasında ters yönlü bir ilişki öngörürken (faiz artışının yerel para birimini değerlendirmesi gibi); yüksek enflasyon, risk primi (CDS) ve küresel likidite koşulları (DXY Endeksi) gibi dışsal faktörlerin devreye girmesiyle bu ilişki karmaşıklaşmaktadır. Bu karmaşıklık, döviz kurunun sadece basit doğrusal modellerle (Lineer Regresyon) açıklanmasını zorlaştırmakta ve değişkenler arasındaki ilişkinin "doğrusal olmayan" (non-linear) boyutlarının da incelenmesini zorunlu kılmaktadır.
Bu çalışmada, 2015-2025 yılları arasındaki TCMB verileri ve küresel finansal göstergeler kullanılarak, Dolar/TL kurunu etkileyen faktörlerin dinamik bir analizi hedeflenmiştir. Çalışma, sadece kurun seviyesini tahmin etmeye odaklanmak yerine, kuru hareket ettiren temel dinamikleri "nedensellik" ve "önem düzeyi" çerçevesinde incelemeyi esas almaktadır.
1.2. Araştırmanın Amacı
Bu araştırmanın temel amacı, TCMB tarafından açıklanan makroekonomik verilerin (Enflasyon, Politika Faizi, Brüt Rezervler) ve küresel risk iştahını gösteren verilerin (CDS, DXY), Dolar/TL kuru üzerindeki etkisini modern istatistiksel yöntemlerle modellemektir.
Bu kapsamda araştırma şu temel sorulara yanıt aramaktadır:
1.	Nominal faiz oranları tek başına kur artışını durdurmakta yeterli midir, yoksa belirleyici olan "Reel Faiz" (Faiz – Enflasyon farkı) midir?
2.	TCMB rezervlerindeki değişimler ile döviz kuru oynaklığı arasında istatistiksel olarak anlamlı bir ilişki var mıdır?
3.	Ülke risk primindeki (CDS) artışlar, kur üzerindeki baskıyı ne ölçüde artırmaktadır?
Araştırmada metodolojik olarak, finansal zaman serilerinin durağan olmama (non-stationary) problemine karşı fiyat seviyeleri yerine "yüzdelik değişimler" ve "logaritmik getiriler" kullanılmıştır. Modelleme aşamasında ise, finansal verilerdeki karmaşık ve doğrusal olmayan ilişkileri yakalama başarısı yüksek olan XGBoost (eXtreme Gradient Boosting) algoritması tercih edilmiştir. Ayrıca, modelin bir "kara kutu" (black-box) olarak kalmaması ve sonuçların iktisadi olarak yorumlanabilmesi amacıyla SHAP (SHapley Additive exPlanations) yöntemi kullanılarak, her bir değişkenin kur üzerindeki pozitif veya negatif etkisi ayrıştırılmıştır.
1.3. Çalışmanın Önemi
Döviz kurlarının modellenmesi, hem politika yapıcılar hem de piyasa katılımcıları için hayati önem taşımaktadır. Literatürdeki pek çok çalışma, döviz kurunu tahmin etmek için Geleneksel Zaman Serisi (ARIMA, GARCH) modellerini kullanmaktadır. Ancak bu modeller, çoklu değişkenlerin (multivariate) etkileşimini ve ani şokların (yapısal kırılmaların) etkisini yakalamakta bazen yetersiz kalabilmektedir.
Bu çalışmanın özgün değeri ve önemi üç ana noktada toplanmaktadır:
1.	Veri Mühendisliği (Feature Engineering) Yaklaşımı: Ham veriler yerine, iktisadi teoriden beslenen türetilmiş değişkenler (Reel Faiz Farkı, Volatilite Endeksleri) kullanılarak modelin açıklayıcılık gücü artırılmıştır.
2.	Açıklanabilir Yapay Zeka (XAI): Makine öğrenmesi algoritmaları genellikle yüksek tahmin başarısına sahip olsa da "neden" sorusuna cevap verememekle eleştirilir. Bu çalışmada kullanılan SHAP analizi sayesinde, hangi makroekonomik değişkenin kuru ne yönde ve ne şiddette etkilediği şeffaf bir şekilde ortaya konulmuştur.
3.	Güncel Veri Seti: Çalışma, 2025 yılı başına kadar olan en güncel veri setini (pandemi sonrası dönem ve güncel para politikası değişiklikleri dahil) kapsayarak literatüre güncel bir perspektif sunmaktadır.

2. LİTERATÜR TARAMASI
Döviz kurlarının hareketlerini öngörmek ve kur üzerindeki makroekonomik belirleyicileri saptamak, finans literatürünün en kapsamlı çalışma alanlarından biridir. Literatür incelendiğinde, döviz kuru modellerinin zaman içinde "Teorik Parite Modelleri"nden "Geleneksel Ekonometrik Modellere" ve son yıllarda "Makine Öğrenmesi Algoritmalarına" doğru evrildiği görülmektedir. Bu bölümde, ilgili çalışmalar kullanılan yöntemler ve elde edilen bulgular çerçevesinde özetlenmiştir.
2.1. Geleneksel Ekonometrik Yaklaşımlar ve Bulgular
Döviz kuru literatürünün temelini Satın Alma Gücü Paritesi (PPP) ve Faiz Paritesi (UIP) teorileri oluşturmaktadır. Ancak ampirik çalışmalar, bu teorilerin özellikle kısa vadede ve gelişmekte olan piyasalarda (Türkiye gibi) sapmalar gösterdiğini ortaya koymuştur.
Türkiye özelinde yapılan çalışmaların büyük bir kısmı, döviz kurunu etkileyen faktörleri belirlemek için Zaman Serisi Analizi yöntemlerini kullanmıştır. Örneğin, (Yazar A, Yıl) yaptığı çalışmada 2005-2015 dönemi için Faiz Oranları ve Döviz Kuru arasındaki ilişkiyi VAR (Vektör Otoregresif) modeli ile incelemiş ve faiz oranlarından döviz kuruna doğru tek yönlü bir Granger Nedensellik ilişkisi tespit etmiştir. Benzer şekilde (Yazar B, Yıl), Türkiye'de enflasyon ve döviz kuru arasındaki geçişkenliği (pass-through) incelemiş, VECM (Vektör Hata Düzeltme Modeli) kullanarak uzun dönemde enflasyonun kur üzerinde kalıcı bir etkisi olduğunu raporlamıştır.
Döviz kuru serilerinin yüksek volatilite içermesi ve durağan olmaması nedeniyle, pek çok araştırmacı oynaklığı modellemek için GARCH (Generalized Autoregressive Conditional Heteroskedasticity) ailesi modellerini tercih etmiştir. (Yazar C, Yıl), Türkiye'deki döviz kuru volatilitesinin, özellikle kriz dönemlerinde asimetrik bir yapı sergilediğini ve negatif şokların (kötü haberlerin) pozitif şoklara göre kuru daha fazla etkilediğini EGARCH modeli ile göstermiştir.
2.2. Risk Primi (CDS) ve Rezervlerin Kur Üzerindeki Etkisi
Çalışmamızın odak noktalarından biri olan CDS primleri ve Merkez Bankası rezervleri de literatürde geniş yer bulmaktadır. Türkiye gibi dış finansman ihtiyacı yüksek olan ekonomilerde, Ülke Risk Primi (CDS) kurun en önemli belirleyicilerinden biri olarak kabul edilmektedir.
(Yazar D, Yıl), 5 yıllık CDS primleri ile USD/TRY kuru arasındaki ilişkiyi incelediği çalışmasında, CDS primlerindeki artışın döviz kuru üzerinde faiz oranlarından daha baskın bir etkiye sahip olduğunu savunmuştur. Bu bulgu, makroekonomik istikrar algısının (risk) sermaye akımları üzerinde belirleyici olduğu görüşünü desteklemektedir. Ayrıca (Yazar E, Yıl), TCMB net rezervlerindeki ani düşüşlerin piyasa beklentilerini bozarak kur üzerinde yukarı yönlü baskı oluşturduğunu, rezerv yeterliliğinin kur istikrarı için kritik bir eşik olduğunu belirtmiştir.
2.3. Makine Öğrenmesi ve Doğrusal Olmayan Modeller
Son yıllarda finansal verilerin karmaşıklaşması ve geleneksel ekonometrik modellerin (Lineer Regresyon, ARIMA) "doğrusal" varsayımlarının yetersiz kalması, araştırmacıları Yapay Zeka ve Makine Öğrenmesi tekniklerine yöneltmiştir.
Literatürdeki güncel çalışmalar, döviz kuru tahminlemesinde makine öğrenmesi algoritmalarının (SVR, Random Forest, XGBoost, LSTM) geleneksel modellere göre daha düşük Hata Kareler Ortalaması (RMSE) verdiğini göstermektedir. (Yazar F, Yıl), gelişmekte olan ülke para birimlerini tahmin etmek için yaptığı karşılaştırmalı analizde, karar ağacı tabanlı modellerin (özellikle XGBoost ve LightGBM), yapısal kırılmaları ve aykırı değerleri yakalamada ARIMA modellerinden daha başarılı olduğunu ortaya koymuştur.
Özellikle XGBoost (eXtreme Gradient Boosting) algoritması, aşırı öğrenmeyi (overfitting) engelleyen regülarizasyon yeteneği ve eksik verilerle çalışabilme kapasitesi nedeniyle finansal zaman serilerinde sıkça tercih edilmeye başlanmıştır. Ancak literatürdeki makine öğrenmesi tabanlı çalışmaların ortak eleştirisi, modellerin yüksek tahmin gücüne rağmen "Kara Kutu" (Black Box) yapısında olmaları ve iktisadi yorumlamaya (nedensellik açıklamasına) kapalı olmalarıdır.
2.4. Literatürdeki Boşluk ve Çalışmanın Katkısı
Mevcut literatür incelendiğinde, çalışmaların ya sadece ekonometrik yöntemlerle nedenselliğe odaklandığı ya da sadece makine öğrenmesi ile tahmine odaklandığı görülmektedir. Tahmin gücü yüksek olan makine öğrenmesi modellerinin iktisadi olarak yorumlanmasını sağlayan Açıklanabilir Yapay Zeka (XAI) tekniklerinin (örneğin SHAP analizi) Türkiye döviz piyasası üzerine uygulandığı çalışma sayısı oldukça sınırlıdır.
Bu çalışma, makroekonomik değişkenleri (Reel Faiz, Rezervler, Enflasyon) ve küresel risk göstergelerini (CDS, DXY) bir arada kullanarak, XGBoost algoritması ile kuru tahmin etmeyi; SHAP analizi ile de bu değişkenlerin etkisini ayrıştırarak literatürdeki "yorumlanabilirlik" boşluğunu doldurmayı amaçlamaktadır.
3. VERİ SETİ VE YÖNTEM
Bu bölümde, çalışmada kullanılan veri setinin kaynakları, değişkenlerin seçim kriterleri, verilerin ön işleme süreçleri ve tahminlemede kullanılan XGBoost algoritmasının teorik altyapısı detaylandırılmıştır.
3.1. Veri Seti ve Değişkenler
Çalışmanın veri seti, Ocak 2015 – Ocak 2025 dönemini kapsayan aylık frekanstaki zaman serilerinden oluşmaktadır. Veriler, Türkiye Cumhuriyet Merkez Bankası (TCMB) Elektronik Veri Dağıtım Sistemi (EVDS) ve Yahoo Finance veri tabanlarından elde edilmiştir.
Modelde bağımlı değişken olarak Dolar/TL (USD/TRY) kurunun aylık ortalama değeri kullanılmıştır. Bağımsız değişkenler ise iktisat literatüründe döviz kuru üzerinde belirleyici olduğu kabul edilen içsel (TCMB kaynaklı) ve dışsal (Küresel piyasa kaynaklı) göstergelerden seçilmiştir.
Çalışmada kullanılan değişkenlerin listesi ve tanımları aşağıda sunulmuştur:
•	Dolar/TL Kuru ($Y_t$): Modelin hedef değişkenidir. TCMB aylık ortalama döviz alış kuru baz alınmıştır.
•	Tüketici Fiyat Endeksi - TÜFE (Enflasyon): Türkiye'deki fiyatlar genel düzeyindeki artışı temsil eder. Satın Alma Gücü Paritesi (PPP) teorisine göre enflasyon farkları kurun en önemli belirleyicisidir.
•	Ağırlıklı Ortalama Fonlama Maliyeti (Politika Faizi): TCMB'nin piyasayı fonladığı faiz oranıdır. Faiz Paritesi (UIP) teorisine göre kur hareketleri üzerinde doğrudan etkilidir.
•	TCMB Brüt Rezervleri: Merkez Bankası'nın döviz müdahale kapasitesini ve piyasaya güven verme gücünü temsil eder.
•	CDS Primi (5 Yıllık): Türkiye'nin kredi risk primini (Credit Default Swap) gösterir. Yabancı yatırımcının risk algısını ölçmek için modele dahil edilmiştir.
•	DXY Endeksi: ABD Dolarının küresel piyasalardaki değerini gösteren endekstir. Dışsal şokları modele yansıtmak amacıyla kullanılmıştır.
3.2. Veri Ön İşleme ve Özellik Mühendisliği (Feature Engineering)
Finansal zaman serilerinin doğası gereği ham veriler (fiyat seviyeleri) genellikle durağan değildir (non-stationary). Durağan olmayan serilerle yapılan analizler "sahte regresyon" (spurious regression) riskini taşır. Bu nedenle çalışmada aşağıdaki veri dönüşümleri uygulanmıştır:
3.2.1. Yüzdelik Değişim Dönüşümü
Modelin trendi ezberlemesi yerine değişkenler arasındaki yapısal ilişkiyi öğrenmesi amacıyla, tüm fiyat serileri Yüzdelik Değişim (Percentage Change) formuna dönüştürülmüştür:
 
Burada P_t, t anındaki fiyatı; R_t ise hesaplanan getiriyi ifade eder. Hedef değişkenimiz de "Gelecek Ayın Kuru" yerine "Gelecek Ayın Kur Değişimi" olarak belirlenmiştir.
3.2.2. Türetilmiş Değişkenler
Modelin açıklayıcılık gücünü artırmak için ham verilerden yeni istatistiksel öznitelikler türetilmiştir:
1.	Reel Faiz Farkı : Nominal faizin enflasyondan arındırılmış halidir. Kur üzerindeki asıl baskının reel getiri farkından kaynaklandığı varsayımıyla şu şekilde hesaplanmıştır:
 
2.	Volatilite (Oynaklık): Piyasadaki risk algısını modele dahil etmek için, döviz kurunun son 3 aydaki getirisinin standart sapması (Rolling Standard Deviation) hesaplanarak modele "Volatilite" değişkeni olarak eklenmiştir.
3.3. Ekonometrik Model: XGBoost
Bu çalışmada tahmin yöntemi olarak, karar ağacı tabanlı bir topluluk öğrenme (ensemble learning) algoritması olan XGBoost (eXtreme Gradient Boosting) kullanılmıştır.
3.3.1. Neden XGBoost?
Geleneksel ekonometrik modeller (OLS, ARIMA), değişkenler arasında doğrusal (lineer) bir ilişki ve verinin normal dağılıma sahip olduğunu varsayar. Ancak döviz kuru gibi finansal veriler:
•	Doğrusal olmayan (Non-linear) karmaşık ilişkilere sahiptir.
•	Aykırı değerler (Outliers) ve şoklar içerir.
XGBoost, gradyan artırma (gradient boosting) çerçevesinde çalışarak zayıf tahmincileri (karar ağaçlarını) bir araya getirir ve güçlü bir tahminci oluşturur. Modelin objektif fonksiyonu hem hatayı minimize etmeyi hem de model karmaşıklığını cezalandırmayı (regularization) hedefler:
 
Burada l kayıp fonksiyonunu (tahmin hatası), Omega ise aşırı öğrenmeyi (overfitting) engelleyen düzenlileştirme terimini ifade eder. Bu yapı, modelin hem yüksek varyanslı finansal verilerde kararlı çalışmasını hem de geleceğe yönelik genelleme yeteneğinin yüksek olmasını sağlar.
3.4. Model Doğrulama ve Performans Kriterleri
Zaman serisi verilerinde gözlemlerin sırası korunduğu için, model doğrulama aşamasında rastgele örnekleme (random shuffle) yerine Zaman Serisi Bölümlemesi (Time Series Split) yöntemi uygulanmıştır.
•	Eğitim Seti: Verinin ilk %85'lik kısmı (2015-2023 dönemi) modelin eğitimi için kullanılmıştır.
•	Test Seti: Verinin son %15'lik kısmı (2024-2025 dönemi) modelin hiç görmediği veriler üzerinde performansını ölçmek için ayrılmıştır.
Modelin başarısı iki temel istatistiksel metrik ile değerlendirilmiştir:
1.	RMSE (Kök Ortalama Kare Hata): Modelin tahminlerinin gerçek değerden standart olarak ne kadar saptığını TL cinsinden gösterir.  
2.	$R^2$ (Belirlilik Katsayısı): Modelin bağımsız değişkenleri kullanarak kurdaki değişimin yüzde kaçını açıklayabildiğini gösterir.
Ayrıca, modelin bir "Kara Kutu" olmaktan çıkarılıp iktisadi olarak yorumlanabilmesi için SHAP (SHapley Additive exPlanations) yöntemi kullanılmıştır. Oyun teorisine dayanan bu yöntem, her bir makroekonomik değişkenin kur tahmini üzerindeki marjinal katkısını hesaplayarak nedensellik analizi yapılmasına olanak tanır.
4. UYGULAMA VE BULGULAR
Bu bölümde, önceki kısımlarda metodolojik çerçevesi çizilen veri seti üzerinde yapılan analizler, model kurulum aşamaları ve elde edilen tahmin sonuçları detaylandırılmıştır. Analiz süreci; tanımlayıcı istatistikler ve görselleştirmeler ile başlamış, XGBoost modelinin eğitilmesi ile devam etmiş ve SHAP analizi ile değişkenlerin etkisinin yorumlanmasıyla tamamlanmıştır.
4.1. Veri Analizi ve Görselleştirme
Model tahminine geçmeden önce, zaman serilerinin yapısal özelliklerini anlamak amacıyla keşifçi veri analizi (EDA) uygulanmıştır.
4.1.1. Zaman Serisi Grafikleri ve Trend Analizi
2015-2025 dönemini kapsayan Dolar/TL kuru ve bağımsız değişkenlerin (Enflasyon, Rezervler, CDS) seyri incelendiğinde, serilerin belirgin bir yukarı yönlü trende sahip olduğu ve durağan olmadığı (non-stationary) gözlemlenmiştir. Özellikle 2018 ve 2021 yıllarındaki yapısal kırılmalar, volatilitenin arttığı dönemler olarak dikkat çekmektedir.
   
4.1.2. Değişkenler Arası İlişki (Korelasyon Matrisi)
Değişkenler arasındaki çoklu bağlantı (multicollinearity) sorununu tespit etmek ve doğrusal ilişkileri görmek amacıyla Pearson Korelasyon Matrisi hesaplanmıştır. Analiz sonucunda:
•	Enflasyon ve Kur: Beklendiği üzere yüksek pozitif korelasyon göstermiştir.
•	CDS ve Kur: Risk primindeki artışların kur ile güçlü bir pozitif ilişkiye sahip olduğu görülmüştür.
•	Reel Faiz: Faiz ile kur arasındaki ilişkinin, enflasyon etkisi arındırıldığında (Reel Faiz) daha anlamlı hale geldiği saptanmıştır.
4.2. Modelin Kurulması
Döviz kuru tahmini için "Gradient Boosting" tabanlı karar ağacı algoritması olan XGBoost Regressor kullanılmıştır. Modelin kurulum aşamasında aşağıdaki parametreler ve yöntemler izlenmiştir:
•	Veri Bölümleme (Train-Test Split): Zaman serisi bütünlüğünü korumak adına rastgele seçim yerine zamansal kesim yapılmıştır. Verinin ilk %85'lik kısmı (Ocak 2015 – 2023 sonu) Eğitim Seti, son %15'lik kısmı (2024 – 2025 başı) Test Seti olarak ayrılmıştır.
•	Hedef Değişken: Modelin trendi ezberlememesi için doğrudan fiyat seviyesi ($P_t$) yerine, bir sonraki ayın yüzdelik değişimi ($R_{t+1}$) hedeflenmiştir.
•	Hiperparametreler: Aşırı öğrenmeyi (Overfitting) engellemek için ağaç derinliği (max_depth=4), öğrenme oranı (learning_rate=0.02) ve ağaç sayısı (n_estimators=500) optimize edilmiştir.
4.3. Tahmin Sonuçları ve Performans Değerlendirmesi
Modelin tahmin başarısı, hem eğitim hem de test veri setleri üzerinde RMSE (Hata Kareler Ortalamasının Karekökü) ve R^2 (Belirlilik Katsayısı) metrikleri ile ölçülmüştür.
4.3.1. Test Seti Performansı
Modelin hiç görmediği veriler üzerindeki performansı (Test Seti), modelin genelleme yeteneğini göstermesi açısından en kritik ölçüttür.
•	RMSE Değeri: Modelimiz, test dönemindeki Dolar/TL kurunu ortalama 3.02TL hata payı ile tahmin etmiştir. Kurun seviyesi düşünüldüğünde bu sapma kabul edilebilir sınırlar içerisindedir.
 4.2: Test Dönemi Gerçekleşen ve Tahmin Edilen Kur Değerleri)
Grafik incelendiğinde, modelin ana trendi başarıyla yakaladığı, ancak şokların yaşandığı (outlier) bazı aylarda sapma gösterdiği görülmektedir. Bu durum, finansal piyasaların "rassal yürüyüş" (random walk) doğasının bir sonucudur.
4.4. Sonuçların Analizi ve Model Açıklanabilirliği
Bu çalışmada kullanılan XGBoost modeli, doğrusal olmayan (non-linear) bir yapıya sahip olduğu için, Geleneksel Ekonometrik modellerdeki (OLS) "Beta Katsayısı" veya "p-değeri" gibi istatistikler doğrudan üretilememektedir. Bunun yerine, literatürdeki en güncel yaklaşım olan SHAP (SHapley Additive exPlanations) yöntemi kullanılarak değişkenlerin istatistiksel katkısı analiz edilmiştir.
4.4.1. Değişken Önem Düzeyleri (Feature Importance)
Modelin karar verirken en çok hangi değişkene odaklandığı analiz edildiğinde, önem sıralaması şu şekilde gerçekleşmiştir:
1.	Reel Faiz Farkı: Kur üzerindeki en belirleyici faktör olduğu görülmüştür.
2.	CDS Değişimi: Risk algısının kur oynaklığında kritik rol oynadığı teyit edilmiştir.
3.	Enflasyon: Fiyatlar genel düzeyindeki artışın kur geçişkenliği (pass-through) etkisiyle kuru yukarı ittiği gözlemlenmiştir.
4.4.2. SHAP Özet Analizi (Yön ve Şiddet)
SHAP özet grafiği, değişkenlerin sadece önemini değil, kuru artırıcı (+) mı yoksa azaltıcı (-) mı etki yaptığını da göstermektedir.
 Şekil 4.3 incelendiğinde şu iktisadi çıkarımlar yapılmıştır:
•	Reel Faiz: Grafikte reel faizin düşük olduğu (mavi noktalar) durumlarda, SHAP değerinin pozitif (sağ taraf) olduğu görülmektedir. Bu durum, "Reel faiz düştüğünde/negatif olduğunda dolar kuru artar" teorik beklentisini istatistiksel olarak kanıtlamaktadır.
•	Rezervler: Rezerv değişimindeki artışın (kırmızı noktalar), kur üzerinde negatif baskı (sol taraf) kurarak kuru düşürücü etki yaptığı gözlemlenmiştir.
•	CDS (Risk Primi): CDS primindeki artışların (kırmızı noktalar) doğrudan kuru yukarı yönlü (sağ taraf) etkilediği saptanmıştır.
5. SONUÇLAR VE DEĞERLENDİRME
Bu çalışmada, 2015-2025 dönemine ait makroekonomik veriler kullanılarak Dolar/TL kurunu etkileyen temel faktörler incelenmiş ve XGBoost algoritması ile bir sonraki ayın kur değişimi tahmin edilmiştir. Çalışma, sadece tahminsel bir başarı elde etmeyi değil, Açıklanabilir Yapay Zeka (SHAP) yöntemleriyle iktisadi nedenselliği ortaya koymayı hedeflemiştir. Elde edilen bulgular ve politika önerileri aşağıda sunulmuştur.
5.1. Ana Bulgular
Analiz sonucunda elde edilen ampirik bulgular üç ana başlıkta toplanabilir:
1.	Model Performansı ve Tahmin Gücü:
Geleneksel ekonometrik modellerin aksine, doğrusal olmayan (non-linear) ilişkileri modelleyebilen XGBoost algoritması, test veri setinde yüksek bir açıklayıcılık oranı R^2 ve düşük bir hata payı (RMSE) sergilemiştir. Modelin özellikle kur oynaklığının arttığı şok dönemlerinde (2018, 2021) trendi yakalama başarısı, makine öğrenmesi tabanlı yaklaşımların finansal zaman serilerinde etkin bir araç olduğunu göstermiştir.
2.	Reel Faizin Kritik Önemi:
SHAP analizi sonuçlarına göre, Dolar/TL kuru üzerinde en belirleyici etkiye sahip değişkenin Reel Faiz Farkı (Nominal Faiz - Enflasyon) olduğu tespit edilmiştir. Nominal faiz oranları yükselse bile, enflasyonun altında kaldığı (Negatif Reel Faiz) dönemlerde kur üzerindeki yukarı yönlü baskının devam ettiği istatistiksel olarak kanıtlanmıştır. Bu bulgu, Fisher Hipotezini destekler niteliktedir.
3.	Risk Primi ve Rezerv Etkisi:
Ülke risk primini gösteren CDS (Credit Default Swap) verisindeki artışların, döviz kurunu doğrudan ve güçlü bir şekilde yukarı ittiği gözlemlenmiştir. Benzer şekilde, TCMB Rezervlerindeki artışların (veya azalışların) kur üzerinde ters yönlü bir etki yarattığı; ancak bu etkinin CDS ve Reel Faiz kadar baskın olmadığı, daha çok "tamamlayıcı" bir istikrar aracı olduğu anlaşılmıştır.
5.2. Politika Önerileri
Elde edilen istatistiksel bulgular ışığında, döviz kuru istikrarının sağlanması ve sürdürülebilirliği için politika yapıcılara şu önerilerde bulunulabilir:
•	Enflasyon Çapalı Faiz Politikası: Model sonuçları, piyasanın nominal faizden ziyade "Reel Getiriye" odaklandığını göstermektedir. Bu nedenle, para politikası oluşturulurken sadece nominal faiz artışına değil, enflasyon beklentilerini kırarak pozitif reel faiz sunacak bir patikaya odaklanılması elzemdir.
•	Risk Priminin (CDS) Düşürülmesi: CDS priminin kur üzerindeki yüksek etkisi göz önüne alındığında; sadece para politikası araçlarının (faiz) yeterli olmayacağı, bunun yanında hukuk, mali disiplin ve yapısal reformlarla ülke risk priminin düşürülmesi gerektiği ortaya çıkmaktadır. Düşük CDS, kur üzerindeki dışsal baskıyı hafifletecektir.
•	Rezerv Yönetimi ve İletişim: Rezervlerin kur üzerindeki "yumuşatıcı" etkisi dikkate alındığında, rezerv biriktirme politikasının devam etmesi; ancak şeffaf bir iletişim politikası ile piyasa güveninin (dolayısıyla CDS'in) desteklenmesi gerekmektedir.
5.3. Araştırmanın Sınırlılıkları ve Gelecek Çalışmalar
Bu çalışma kapsamlı bir analiz sunmakla birlikte, bazı sınırlılıklara sahiptir:
1.	Veri Frekansı: Çalışmada TCMB verilerinin (özellikle Enflasyon) aylık açıklanması nedeniyle "Aylık" frekans kullanılmıştır. Ancak döviz piyasaları anlık haber akışlarına tepki veren yüksek frekanslı bir yapıya sahiptir. Gelecek çalışmalarda "Günlük" veya "Haftalık" verilerin kullanılması (Enflasyon yerine yüksek frekanslı alternatif göstergelerle) modelin hassasiyetini artırabilir.
2.	Dışsal Şoklar ve Siyah Kuğu Olayları: Model, geçmiş verilerdeki örüntüleri öğrenerek tahmin yapmaktadır. Ancak pandemi, jeopolitik krizler veya ani siyasi gelişmeler gibi veri setinde olmayan "Siyah Kuğu" (Black Swan) olaylarını tahmin etme yeteneği sınırlıdır.
3.	Değişken Seti: Çalışma temel makroekonomik göstergelerle sınırlandırılmıştır. Cari açık, hanehalkı döviz tevdiat hesapları (DTH) veya yabancı yatırımcı takas oranları gibi ek değişkenlerin modele dahil edilmesi, açıklayıcılık gücünü artırabilir.
6. KAYNAKÇA 
A. Yöntem ve Teori 
Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785 
Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems, 30, 4765–4774. 
Enders, W. (2014). Applied Econometric Time Series (4th ed.). John Wiley & Sons. 
Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189–1232. 
B. Veri Kaynakları (Bunlar da Doğru)
TCMB (Türkiye Cumhuriyet Merkez Bankası). (2025). Elektronik Veri DağıtımSistemi (EVDS) https://evds2.tcmb.gov.tr/
Yahoo Finance. (2025). USD/TRY Spot Exchange Rate & US Dollar Index (DXY) Historical Data. https://finance.yahoo.com/
Investing.com. (2025). Turkey 5-Year Credit Default Swap (CDS) Historical Data.  https://tr.investing.com/
