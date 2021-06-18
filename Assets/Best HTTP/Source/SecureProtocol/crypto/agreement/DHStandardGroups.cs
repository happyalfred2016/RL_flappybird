#if !BESTHTTP_DISABLE_ALTERNATE_SSL && (!UNITY_WEBGL || UNITY_EDITOR)
#pragma warning disable
using System;

using BestHTTP.SecureProtocol.Org.BouncyCastle.Crypto.Parameters;
using BestHTTP.SecureProtocol.Org.BouncyCastle.Math;
using BestHTTP.SecureProtocol.Org.BouncyCastle.Utilities.Encoders;

namespace BestHTTP.SecureProtocol.Org.BouncyCastle.Crypto.Agreement
{
    /// <summary>Standard Diffie-Hellman groups from various IETF specifications.</summary>
    public class DHStandardGroups
    {
        private static BigInteger FromHex(string hex)
        {
            return new BigInteger(1, Hex.DecodeStrict(hex));
        }

        private static DHParameters FromPG(string hexP, string hexG)
        {
            return new DHParameters(FromHex(hexP), FromHex(hexG));
        }

        private static DHParameters FromPGQ(string hexP, string hexG, string hexQ)
        {
            return new DHParameters(FromHex(hexP), FromHex(hexG), FromHex(hexQ));
        }

        private static DHParameters Rfc7919Parameters(string hexP, int l)
        {
            // NOTE: All the groups in RFC 7919 use safe primes, i.e. q = (p-1)/2, and generator g = 2
            BigInteger p = FromHex(hexP);
            return new DHParameters(p, BigInteger.Two, p.ShiftRight(1), l);
        }

        /*
         * RFC 2409
         */
        private static readonly string rfc2409_768_p = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
            + "29024E088A67CC74020BBEA63B139B22514A08798E3404DD" + "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
            + "E485B576625E7EC6F44C42E9A63A3620FFFFFFFFFFFFFFFF";
        private static readonly string rfc2409_768_g = "02";
        public static readonly DHParameters rfc2409_768 = FromPG(rfc2409_768_p, rfc2409_768_g);

        private static readonly string rfc2409_1024_p = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
            + "29024E088A67CC74020BBEA63B139B22514A08798E3404DD" + "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
            + "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED" + "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE65381"
            + "FFFFFFFFFFFFFFFF";
        private static readonly string rfc2409_1024_g = "02";
        public static readonly DHParameters rfc2409_1024 = FromPG(rfc2409_1024_p, rfc2409_1024_g);

        /*
         * RFC 3526
         */
        private static readonly string rfc3526_1536_p = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
            + "29024E088A67CC74020BBEA63B139B22514A08798E3404DD" + "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
            + "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED" + "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
            + "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F" + "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
            + "670C354E4ABC9804F1746C08CA237327FFFFFFFFFFFFFFFF";
        private static readonly string rfc3526_1536_g = "02";
        public static readonly DHParameters rfc3526_1536 = FromPG(rfc3526_1536_p, rfc3526_1536_g);

        private static readonly string rfc3526_2048_p = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
            + "29024E088A67CC74020BBEA63B139B22514A08798E3404DD" + "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
            + "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED" + "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
            + "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F" + "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
            + "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B" + "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
            + "DE2BCBF6955817183995497CEA956AE515D2261898FA0510" + "15728E5A8AACAA68FFFFFFFFFFFFFFFF";
        private static readonly string rfc3526_2048_g = "02";
        public static readonly DHParameters rfc3526_2048 = FromPG(rfc3526_2048_p, rfc3526_2048_g);

        private static readonly string rfc3526_3072_p = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
            + "29024E088A67CC74020BBEA63B139B22514A08798E3404DD" + "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
            + "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED" + "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
            + "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F" + "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
            + "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B" + "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
            + "DE2BCBF6955817183995497CEA956AE515D2261898FA0510" + "15728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64"
            + "ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7" + "ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6B"
            + "F12FFA06D98A0864D87602733EC86A64521F2B18177B200C" + "BBE117577A615D6C770988C0BAD946E208E24FA074E5AB31"
            + "43DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF";
        private static readonly string rfc3526_3072_g = "02";
        public static readonly DHParameters rfc3526_3072 = FromPG(rfc3526_3072_p, rfc3526_3072_g);

        private static readonly string rfc3526_4096_p = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
            + "29024E088A67CC74020BBEA63B139B22514A08798E3404DD" + "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
            + "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED" + "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
            + "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F" + "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
            + "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B" + "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
            + "DE2BCBF6955817183995497CEA956AE515D2261898FA0510" + "15728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64"
            + "ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7" + "ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6B"
            + "F12FFA06D98A0864D87602733EC86A64521F2B18177B200C" + "BBE117577A615D6C770988C0BAD946E208E24FA074E5AB31"
            + "43DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D7" + "88719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA"
            + "2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6" + "287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED"
            + "1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA9" + "93B4EA988D8FDDC186FFB7DC90A6C08F4DF435C934063199"
            + "FFFFFFFFFFFFFFFF";
        private static readonly string rfc3526_4096_g = "02";
        public static readonly DHParameters rfc3526_4096 = FromPG(rfc3526_4096_p, rfc3526_4096_g);

        private static readonly string rfc3526_6144_p = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E08"
            + "8A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B"
            + "302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9"
            + "A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE6"
            + "49286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8"
            + "FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D"
            + "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C"
            + "180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
            + "3995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D"
            + "04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7D"
            + "B3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D226"
            + "1AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200C"
            + "BBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFC"
            + "E0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B26"
            + "99C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB"
            + "04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2"
            + "233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127"
            + "D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C934028492"
            + "36C3FAB4D27C7026C1D4DCB2602646DEC9751E763DBA37BDF8FF9406"
            + "AD9E530EE5DB382F413001AEB06A53ED9027D831179727B0865A8918"
            + "DA3EDBEBCF9B14ED44CE6CBACED4BB1BDB7F1447E6CC254B33205151"
            + "2BD7AF426FB8F401378CD2BF5983CA01C64B92ECF032EA15D1721D03"
            + "F482D7CE6E74FEF6D55E702F46980C82B5A84031900B1C9E59E7C97F"
            + "BEC7E8F323A97A7E36CC88BE0F1D45B7FF585AC54BD407B22B4154AA"
            + "CC8F6D7EBF48E1D814CC5ED20F8037E0A79715EEF29BE32806A1D58B"
            + "B7C5DA76F550AA3D8A1FBFF0EB19CCB1A313D55CDA56C9EC2EF29632"
            + "387FE8D76E3C0468043E8F663F4860EE12BF2D5B0B7474D6E694F91E"
            + "6DCC4024FFFFFFFFFFFFFFFF";
        private static readonly string rfc3526_6144_g = "02";
        public static readonly DHParameters rfc3526_6144 = FromPG(rfc3526_6144_p, rfc3526_6144_g);

        private static readonly string rfc3526_8192_p = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
            + "29024E088A67CC74020BBEA63B139B22514A08798E3404DD" + "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
            + "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED" + "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
            + "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F" + "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
            + "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B" + "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
            + "DE2BCBF6955817183995497CEA956AE515D2261898FA0510" + "15728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64"
            + "ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7" + "ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6B"
            + "F12FFA06D98A0864D87602733EC86A64521F2B18177B200C" + "BBE117577A615D6C770988C0BAD946E208E24FA074E5AB31"
            + "43DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D7" + "88719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA"
            + "2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6" + "287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED"
            + "1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA9" + "93B4EA988D8FDDC186FFB7DC90A6C08F4DF435C934028492"
            + "36C3FAB4D27C7026C1D4DCB2602646DEC9751E763DBA37BD" + "F8FF9406AD9E530EE5DB382F413001AEB06A53ED9027D831"
            + "179727B0865A8918DA3EDBEBCF9B14ED44CE6CBACED4BB1B" + "DB7F1447E6CC254B332051512BD7AF426FB8F401378CD2BF"
            + "5983CA01C64B92ECF032EA15D1721D03F482D7CE6E74FEF6" + "D55E702F46980C82B5A84031900B1C9E59E7C97FBEC7E8F3"
            + "23A97A7E36CC88BE0F1D45B7FF585AC54BD407B22B4154AA" + "CC8F6D7EBF48E1D814CC5ED20F8037E0A79715EEF29BE328"
            + "06A1D58BB7C5DA76F550AA3D8A1FBFF0EB19CCB1A313D55C" + "DA56C9EC2EF29632387FE8D76E3C0468043E8F663F4860EE"
            + "12BF2D5B0B7474D6E694F91E6DBE115974A3926F12FEE5E4" + "38777CB6A932DF8CD8BEC4D073B931BA3BC832B68D9DD300"
            + "741FA7BF8AFC47ED2576F6936BA424663AAB639C5AE4F568" + "3423B4742BF1C978238F16CBE39D652DE3FDB8BEFC848AD9"
            + "22222E04A4037C0713EB57A81A23F0C73473FC646CEA306B" + "4BCBC8862F8385DDFA9D4B7FA2C087E879683303ED5BDD3A"
            + "062B3CF5B3A278A66D2A13F83F44F82DDF310EE074AB6A36" + "4597E899A0255DC164F31CC50846851DF9AB48195DED7EA1"
            + "B1D510BD7EE74D73FAF36BC31ECFA268359046F4EB879F92" + "4009438B481C6CD7889A002ED5EE382BC9190DA6FC026E47"
            + "9558E4475677E9AA9E3050E2765694DFC81F56E880B96E71" + "60C980DD98EDD3DFFFFFFFFFFFFFFFFF";
        private static readonly string rfc3526_8192_g = "02";
        public static readonly DHParameters rfc3526_8192 = FromPG(rfc3526_8192_p, rfc3526_8192_g);

        /*
         * RFC 4306
         */
        public static readonly DHParameters rfc4306_768 = rfc2409_768;
        public static readonly DHParameters rfc4306_1024 = rfc2409_1024;

        /*
         * RFC 5114
         */
        private static readonly string rfc5114_1024_160_p = "B10B8F96A080E01DDE92DE5EAE5D54EC52C99FBCFB06A3C6"
            + "9A6A9DCA52D23B616073E28675A23D189838EF1E2EE652C0" + "13ECB4AEA906112324975C3CD49B83BFACCBDD7D90C4BD70"
            + "98488E9C219A73724EFFD6FAE5644738FAA31A4FF55BCCC0" + "A151AF5F0DC8B4BD45BF37DF365C1A65E68CFDA76D4DA708"
            + "DF1FB2BC2E4A4371";
        private static readonly string rfc5114_1024_160_g = "A4D1CBD5C3FD34126765A442EFB99905F8104DD258AC507F"
            + "D6406CFF14266D31266FEA1E5C41564B777E690F5504F213" + "160217B4B01B886A5E91547F9E2749F4D7FBD7D3B9A92EE1"
            + "909D0D2263F80A76A6A24C087A091F531DBF0A0169B6A28A" + "D662A4D18E73AFA32D779D5918D08BC8858F4DCEF97C2A24"
            + "855E6EEB22B3B2E5";
        private static readonly string rfc5114_1024_160_q = "F518AA8781A8DF278ABA4E7D64B7CB9D49462353";

        /// <remarks>
        /// Existence of a "hidden SNFS" backdoor cannot be ruled out. see https://eprint.iacr.org/2016/961.pdf .
        /// </remarks>

        public static readonly DHParameters rfc5114_1024_160 = FromPGQ(rfc5114_1024_160_p, rfc5114_1024_160_g,
            rfc5114_1024_160_q);

        private static readonly string rfc5114_2048_224_p = "AD107E1E9123A9D0D660FAA79559C51FA20D64E5683B9FD1"
            + "B54B1597B61D0A75E6FA141DF95A56DBAF9A3C407BA1DF15" + "EB3D688A309C180E1DE6B85A1274A0A66D3F8152AD6AC212"
            + "9037C9EDEFDA4DF8D91E8FEF55B7394B7AD5B7D0B6C12207" + "C9F98D11ED34DBF6C6BA0B2C8BBC27BE6A00E0A0B9C49708"
            + "B3BF8A317091883681286130BC8985DB1602E714415D9330" + "278273C7DE31EFDC7310F7121FD5A07415987D9ADC0A486D"
            + "CDF93ACC44328387315D75E198C641A480CD86A1B9E587E8" + "BE60E69CC928B2B9C52172E413042E9B23F10B0E16E79763"
            + "C9B53DCF4BA80A29E3FB73C16B8E75B97EF363E2FFA31F71" + "CF9DE5384E71B81C0AC4DFFE0C10E64F";
        private static readonly string rfc5114_2048_224_g = "AC4032EF4F2D9AE39DF30B5C8FFDAC506CDEBE7B89998CAF"
            + "74866A08CFE4FFE3A6824A4E10B9A6F0DD921F01A70C4AFA" + "AB739D7700C29F52C57DB17C620A8652BE5E9001A8D66AD7"
            + "C17669101999024AF4D027275AC1348BB8A762D0521BC98A" + "E247150422EA1ED409939D54DA7460CDB5F6C6B250717CBE"
            + "F180EB34118E98D119529A45D6F834566E3025E316A330EF" + "BB77A86F0C1AB15B051AE3D428C8F8ACB70A8137150B8EEB"
            + "10E183EDD19963DDD9E263E4770589EF6AA21E7F5F2FF381" + "B539CCE3409D13CD566AFBB48D6C019181E1BCFE94B30269"
            + "EDFE72FE9B6AA4BD7B5A0F1C71CFFF4C19C418E1F6EC0179" + "81BC087F2A7065B384B890D3191F2BFA";
        private static readonly string rfc5114_2048_224_q = "801C0D34C58D93FE997177101F80535A4738CEBCBF389A99B36371EB";

        /// <remarks>
        /// Existence of a "hidden SNFS" backdoor cannot be ruled out. see https://eprint.iacr.org/2016/961.pdf .
        /// </remarks>

        public static readonly DHParameters rfc5114_2048_224 = FromPGQ(rfc5114_2048_224_p, rfc5114_2048_224_g,
            rfc5114_2048_224_q);

        private static readonly string rfc5114_2048_256_p = "87A8E61DB4B6663CFFBBD19C651959998CEEF608660DD0F2"
            + "5D2CEED4435E3B00E00DF8F1D61957D4FAF7DF4561B2AA30" + "16C3D91134096FAA3BF4296D830E9A7C209E0C6497517ABD"
            + "5A8A9D306BCF67ED91F9E6725B4758C022E0B1EF4275BF7B" + "6C5BFC11D45F9088B941F54EB1E59BB8BC39A0BF12307F5C"
            + "4FDB70C581B23F76B63ACAE1CAA6B7902D52526735488A0E" + "F13C6D9A51BFA4AB3AD8347796524D8EF6A167B5A41825D9"
            + "67E144E5140564251CCACB83E6B486F6B3CA3F7971506026" + "C0B857F689962856DED4010ABD0BE621C3A3960A54E710C3"
            + "75F26375D7014103A4B54330C198AF126116D2276E11715F" + "693877FAD7EF09CADB094AE91E1A1597";
        private static readonly string rfc5114_2048_256_g = "3FB32C9B73134D0B2E77506660EDBD484CA7B18F21EF2054"
            + "07F4793A1A0BA12510DBC15077BE463FFF4FED4AAC0BB555" + "BE3A6C1B0C6B47B1BC3773BF7E8C6F62901228F8C28CBB18"
            + "A55AE31341000A650196F931C77A57F2DDF463E5E9EC144B" + "777DE62AAAB8A8628AC376D282D6ED3864E67982428EBC83"
            + "1D14348F6F2F9193B5045AF2767164E1DFC967C1FB3F2E55" + "A4BD1BFFE83B9C80D052B985D182EA0ADB2A3B7313D3FE14"
            + "C8484B1E052588B9B7D2BBD2DF016199ECD06E1557CD0915" + "B3353BBB64E0EC377FD028370DF92B52C7891428CDC67EB6"
            + "184B523D1DB246C32F63078490F00EF8D647D148D4795451" + "5E2327CFEF98C582664B4C0F6CC41659";
        private static readonly string rfc5114_2048_256_q = "8CF83642A709A097B447997640129DA299B1A47D1EB3750B"
            + "A308B0FE64F5FBD3";

        /// <remarks>
        /// Existence of a "hidden SNFS" backdoor cannot be ruled out. see https://eprint.iacr.org/2016/961.pdf .
        /// </remarks>

        public static readonly DHParameters rfc5114_2048_256 = FromPGQ(rfc5114_2048_256_p, rfc5114_2048_256_g,
            rfc5114_2048_256_q);

        /*
         * RFC 5996
         */
        public static readonly DHParameters rfc5996_768 = rfc4306_768;
        public static readonly DHParameters rfc5996_1024 = rfc4306_1024;

        /*
         * RFC 7919
         */
        private static readonly string rfc7919_ffdhe2048_p = "FFFFFFFFFFFFFFFFADF85458A2BB4A9AAFDC5620273D3CF1"
            + "D8B9C583CE2D3695A9E13641146433FBCC939DCE249B3EF9" + "7D2FE363630C75D8F681B202AEC4617AD3DF1ED5D5FD6561"
            + "2433F51F5F066ED0856365553DED1AF3B557135E7F57C935" + "984F0C70E0E68B77E2A689DAF3EFE8721DF158A136ADE735"
            + "30ACCA4F483A797ABC0AB182B324FB61D108A94BB2C8E3FB" + "B96ADAB760D7F4681D4F42A3DE394DF4AE56EDE76372BB19"
            + "0B07A7C8EE0A6D709E02FCE1CDF7E2ECC03404CD28342F61" + "9172FE9CE98583FF8E4F1232EEF28183C3FE3B1B4C6FAD73"
            + "3BB5FCBC2EC22005C58EF1837D1683B2C6F34A26C1B2EFFA" + "886B423861285C97FFFFFFFFFFFFFFFF";
        public static readonly DHParameters rfc7919_ffdhe2048 = Rfc7919Parameters(rfc7919_ffdhe2048_p, 225);

        private static readonly string rfc7919_ffdhe3072_p = "FFFFFFFFFFFFFFFFADF85458A2BB4A9AAFDC5620273D3CF1"
            + "D8B9C583CE2D3695A9E13641146433FBCC939DCE249B3EF9" + "7D2FE363630C75D8F681B202AEC4617AD3DF1ED5D5FD6561"
            + "2433F51F5F066ED0856365553DED1AF3B557135E7F57C935" + "984F0C70E0E68B77E2A689DAF3EFE8721DF158A136ADE735"
            + "30ACCA4F483A797ABC0AB182B324FB61D108A94BB2C8E3FB" + "B96ADAB760D7F4681D4F42A3DE394DF4AE56EDE76372BB19"
            + "0B07A7C8EE0A6D709E02FCE1CDF7E2ECC03404CD28342F61" + "9172FE9CE98583FF8E4F1232EEF28183C3FE3B1B4C6FAD73"
            + "3BB5FCBC2EC22005C58EF1837D1683B2C6F34A26C1B2EFFA" + "886B4238611FCFDCDE355B3B6519035BBC34F4DEF99C0238"
            + "61B46FC9D6E6C9077AD91D2691F7F7EE598CB0FAC186D91C" + "AEFE130985139270B4130C93BC437944F4FD4452E2D74DD3"
            + "64F2E21E71F54BFF5CAE82AB9C9DF69EE86D2BC522363A0D" + "ABC521979B0DEADA1DBF9A42D5C4484E0ABCD06BFA53DDEF"
            + "3C1B20EE3FD59D7C25E41D2B66C62E37FFFFFFFFFFFFFFFF";
        public static readonly DHParameters rfc7919_ffdhe3072 = Rfc7919Parameters(rfc7919_ffdhe3072_p, 275);

        private static readonly string rfc7919_ffdhe4096_p = "FFFFFFFFFFFFFFFFADF85458A2BB4A9AAFDC5620273D3CF1"
            + "D8B9C583CE2D3695A9E13641146433FBCC939DCE249B3EF9" + "7D2FE363630C75D8F681B202AEC4617AD3DF1ED5D5FD6561"
            + "2433F51F5F066ED0856365553DED1AF3B557135E7F57C935" + "984F0C70E0E68B77E2A689DAF3EFE8721DF158A136ADE735"
            + "30ACCA4F483A797ABC0AB182B324FB61D108A94BB2C8E3FB" + "B96ADAB760D7F4681D4F42A3DE394DF4AE56EDE76372BB19"
            + "0B07A7C8EE0A6D709E02FCE1CDF7E2ECC03404CD28342F61" + "9172FE9CE98583FF8E4F1232EEF28183C3FE3B1B4C6FAD73"
            + "3BB5FCBC2EC22005C58EF1837D1683B2C6F34A26C1B2EFFA" + "886B4238611FCFDCDE355B3B6519035BBC34F4DEF99C0238"
            + "61B46FC9D6E6C9077AD91D2691F7F7EE598CB0FAC186D91C" + "AEFE130985139270B4130C93BC437944F4FD4452E2D74DD3"
            + "64F2E21E71F54BFF5CAE82AB9C9DF69EE86D2BC522363A0D" + "ABC521979B0DEADA1DBF9A42D5C4484E0ABCD06BFA53DDEF"
            + "3C1B20EE3FD59D7C25E41D2B669E1EF16E6F52C3164DF4FB" + "7930E9E4E58857B6AC7D5F42D69F6D187763CF1D55034004"
            + "87F55BA57E31CC7A7135C886EFB4318AED6A1E012D9E6832" + "A907600A918130C46DC778F971AD0038092999A333CB8B7A"
            + "1A1DB93D7140003C2A4ECEA9F98D0ACC0A8291CDCEC97DCF" + "8EC9B55A7F88A46B4DB5A851F44182E1C68A007E5E655F6A"
            + "FFFFFFFFFFFFFFFF";
        public static readonly DHParameters rfc7919_ffdhe4096 = Rfc7919Parameters(rfc7919_ffdhe4096_p, 325);

        private static readonly string rfc7919_ffdhe6144_p = "FFFFFFFFFFFFFFFFADF85458A2BB4A9AAFDC5620273D3CF1"
            + "D8B9C583CE2D3695A9E13641146433FBCC939DCE249B3EF9" + "7D2FE363630C75D8F681B202AEC4617AD3DF1ED5D5FD6561"
            + "2433F51F5F066ED0856365553DED1AF3B557135E7F57C935" + "984F0C70E0E68B77E2A689DAF3EFE8721DF158A136ADE735"
            + "30ACCA4F483A797ABC0AB182B324FB61D108A94BB2C8E3FB" + "B96ADAB760D7F4681D4F42A3DE394DF4AE56EDE76372BB19"
            + "0B07A7C8EE0A6D709E02FCE1CDF7E2ECC03404CD28342F61" + "9172FE9CE98583FF8E4F1232EEF28183C3FE3B1B4C6FAD73"
            + "3BB5FCBC2EC22005C58EF1837D1683B2C6F34A26C1B2EFFA" + "886B4238611FCFDCDE355B3B6519035BBC34F4DEF99C0238"
            + "61B46FC9D6E6C9077AD91D2691F7F7EE598CB0FAC186D91C" + "AEFE130985139270B4130C93BC437944F4FD4452E2D74DD3"
            + "64F2E21E71F54BFF5CAE82AB9C9DF69EE86D2BC522363A0D" + "ABC521979B0DEADA1DBF9A42D5C4484E0ABCD06BFA53DDEF"
            + "3C1B20EE3FD59D7C25E41D2B669E1EF16E6F52C3164DF4FB" + "7930E9E4E58857B6AC7D5F42D69F6D187763CF1D55034004"
            + "87F55BA57E31CC7A7135C886EFB4318AED6A1E012D9E6832" + "A907600A918130C46DC778F971AD0038092999A333CB8B7A"
            + "1A1DB93D7140003C2A4ECEA9F98D0ACC0A8291CDCEC97DCF" + "8EC9B55A7F88A46B4DB5A851F44182E1C68A007E5E0DD902"
            + "0BFD64B645036C7A4E677D2C38532A3A23BA4442CAF53EA6" + "3BB454329B7624C8917BDD64B1C0FD4CB38E8C334C701C3A"
            + "CDAD0657FCCFEC719B1F5C3E4E46041F388147FB4CFDB477" + "A52471F7A9A96910B855322EDB6340D8A00EF092350511E3"
            + "0ABEC1FFF9E3A26E7FB29F8C183023C3587E38DA0077D9B4" + "763E4E4B94B2BBC194C6651E77CAF992EEAAC0232A281BF6"
            + "B3A739C1226116820AE8DB5847A67CBEF9C9091B462D538C" + "D72B03746AE77F5E62292C311562A846505DC82DB854338A"
            + "E49F5235C95B91178CCF2DD5CACEF403EC9D1810C6272B04" + "5B3B71F9DC6B80D63FDD4A8E9ADB1E6962A69526D43161C1"
            + "A41D570D7938DAD4A40E329CD0E40E65FFFFFFFFFFFFFFFF";
        public static readonly DHParameters rfc7919_ffdhe6144 = Rfc7919Parameters(rfc7919_ffdhe6144_p, 375);

        private static readonly string rfc7919_ffdhe8192_p = "FFFFFFFFFFFFFFFFADF85458A2BB4A9AAFDC5620273D3CF1"
            + "D8B9C583CE2D3695A9E13641146433FBCC939DCE249B3EF9" + "7D2FE363630C75D8F681B202AEC4617AD3DF1ED5D5FD6561"
            + "2433F51F5F066ED0856365553DED1AF3B557135E7F57C935" + "984F0C70E0E68B77E2A689DAF3EFE8721DF158A136ADE735"
            + "30ACCA4F483A797ABC0AB182B324FB61D108A94BB2C8E3FB" + "B96ADAB760D7F4681D4F42A3DE394DF4AE56EDE76372BB19"
            + "0B07A7C8EE0A6D709E02FCE1CDF7E2ECC03404CD28342F61" + "9172FE9CE98583FF8E4F1232EEF28183C3FE3B1B4C6FAD73"
            + "3BB5FCBC2EC22005C58EF1837D1683B2C6F34A26C1B2EFFA" + "886B4238611FCFDCDE355B3B6519035BBC34F4DEF99C0238"
            + "61B46FC9D6E6C9077AD91D2691F7F7EE598CB0FAC186D91C" + "AEFE130985139270B4130C93BC437944F4FD4452E2D74DD3"
            + "64F2E21E71F54BFF5CAE82AB9C9DF69EE86D2BC522363A0D" + "ABC521979B0DEADA1DBF9A42D5C4484E0ABCD06BFA53DDEF"
            + "3C1B20EE3FD59D7C25E41D2B669E1EF16E6F52C3164DF4FB" + "7930E9E4E58857B6AC7D5F42D69F6D187763CF1D55034004"
            + "87F55BA57E31CC7A7135C886EFB4318AED6A1E012D9E6832" + "A907600A918130C46DC778F971AD0038092999A333CB8B7A"
            + "1A1DB93D7140003C2A4ECEA9F98D0ACC0A8291CDCEC97DCF" + "8EC9B55A7F88A46B4DB5A851F44182E1C68A007E5E0DD902"
            + "0BFD64B645036C7A4E677D2C38532A3A23BA4442CAF53EA6" + "3BB454329B7624C8917BDD64B1C0FD4CB38E8C334C701C3A"
            + "CDAD0657FCCFEC719B1F5C3E4E46041F388147FB4CFDB477" + "A52471F7A9A96910B855322EDB6340D8A00EF092350511E3"
            + "0ABEC1FFF9E3A26E7FB29F8C183023C3587E38DA0077D9B4" + "763E4E4B94B2BBC194C6651E77CAF992EEAAC0232A281BF6"
            + "B3A739C1226116820AE8DB5847A67CBEF9C9091B462D538C" + "D72B03746AE77F5E62292C311562A846505DC82DB854338A"
            + "E49F5235C95B91178CCF2DD5CACEF403EC9D1810C6272B04" + "5B3B71F9DC6B80D63FDD4A8E9ADB1E6962A69526D43161C1"
            + "A41D570D7938DAD4A40E329CCFF46AAA36AD004CF600C838" + "1E425A31D951AE64FDB23FCEC9509D43687FEB69EDD1CC5E"
            + "0B8CC3BDF64B10EF86B63142A3AB8829555B2F747C932665" + "CB2C0F1CC01BD70229388839D2AF05E454504AC78B758282"
            + "2846C0BA35C35F5C59160CC046FD8251541FC68C9C86B022" + "BB7099876A460E7451A8A93109703FEE1C217E6C3826E52C"
            + "51AA691E0E423CFC99E9E31650C1217B624816CDAD9A95F9" + "D5B8019488D9C0A0A1FE3075A577E23183F81D4A3F2FA457"
            + "1EFC8CE0BA8A4FE8B6855DFE72B0A66EDED2FBABFBE58A30" + "FAFABE1C5D71A87E2F741EF8C1FE86FEA6BBFDE530677F0D"
            + "97D11D49F7A8443D0822E506A9F4614E011E2A94838FF88C" + "D68C8BB7C5C6424CFFFFFFFFFFFFFFFF";
        public static readonly DHParameters rfc7919_ffdhe8192 = Rfc7919Parameters(rfc7919_ffdhe8192_p, 400);
    }
}
#pragma warning restore
#endif